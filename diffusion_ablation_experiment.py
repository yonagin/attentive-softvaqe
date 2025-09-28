import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from models.soft_vqvae import SoftVQVAE
from pixelcnn.models import GatedPixelCNN
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader

# 引入 Hugging Face Diffusers 库
# pip install diffusers accelerate transformers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

# 图像质量评估指标
# pip install torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# 引入 OrthoVAE 模型
from models.ortho_vae import OrthoVAE


# ==============================================================================
# 1. VQVAE + PixelCNN 部分 (大部分沿用原代码, 稍作调整)
# ==============================================================================

def extract_discrete_latents_vqvae(model, data_loader, device):
    model.eval()
    all_codes = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(data_loader):
            x = x.to(device)
            
            # Get latent codes from VQVAE - both models use encoder attribute
            z = model.encoder(x)
            z = model.pre_quantization_conv(z)
            
            # For VQVAE
            embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = model.vector_quantization(z)
            
            # Use the indices directly from the vector quantizer
            indices = min_encoding_indices.squeeze(1)

            
            # Reshape indices to match latent spatial dimensions
            latent_shape = z.shape[2:]
            indices = indices.view(x.shape[0], *latent_shape)
            
            all_codes.append(indices.cpu())
            all_labels.append(labels)
    
    all_codes = torch.cat(all_codes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_codes, all_labels


def train_pixelcnn(model, model_name, training_loader, validation_loader, args, device):
    """
    在 VQVAE 的离散隐空间上训练 PixelCNN.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"--- Training PixelCNN on {model_name} latent space ---")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(training_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for codes, labels in progress_bar:
            codes = codes.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(codes, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(logits.view(-1, args.n_embeddings), codes.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(training_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
    return model


def generate_with_pixelcnn(pixelcnn_model, vqvae_model, num_samples, latent_shape, device):
    """
    使用 PixelCNN 生成样本, 并用 VQVAE 解码器解码.
    """
    pixelcnn_model.eval()
    vqvae_model.eval()
    
    with torch.no_grad():
        # 1. PixelCNN 生成离散索引
        # GatedPixelCNN的generate方法需要类别标签，这里我们简化为无条件生成
        # 所以传入一个假的labels参数，如果你的GatedPixelCNN支持无条件，请修改
        dummy_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
        generated_codes = pixelcnn_model.generate(dummy_labels, shape=latent_shape, batch_size=num_samples)
        
        # 2. 将索引转换为词向量 (embeddings)
        generated_codes_flat = generated_codes.view(-1)
        embeddings = vqvae_model.vector_quantization.embedding(generated_codes_flat)
        
        # 调整形状以匹配解码器输入
        # embeddings shape: [B*H*W, D] -> [B, H, W, D] -> [B, D, H, W]
        embeddings = embeddings.view(
            num_samples, 
            latent_shape[0], 
            latent_shape[1], 
            vqvae_model.vector_quantization.e_dim
        ).permute(0, 3, 1, 2).contiguous()
        
        # 3. VQVAE 解码器解码成图像
        generated_images = vqvae_model.decoder(embeddings)
    
    return generated_images


# ==============================================================================
# 2. 通用潜向量提取 + Diffusion Model 部分
# ==============================================================================

def extract_continuous_latents(model, model_type, data_loader, device, save_dir=None):
    """
    通用潜向量提取函数，支持 ortho_vae 和 softvqvae 两种模型。
    
    Args:
        model: 预训练的模型
        model_type: 模型类型，'ortho_vae' 或 'softvqvae'
        data_loader: 数据加载器
        device: 设备
        save_dir: 保存目录
    
    Returns:
        standardized_latents: 标准化后的潜向量
        mean: 均值
        std: 标准差
    """
    model.eval()
    all_latents = []
    
    with torch.no_grad():
        for x, _ in tqdm(data_loader, desc=f"Extracting {model_type} Latents"):
            x = x.to(device)
            
            if model_type == 'softvqvae':
                # SoftVQVAE: 提取量化后的潜变量 z_q
                z_e = model.encoder(x)
                z_e = model.pre_quantization_conv(z_e)
                z_q, _ = model.quantizer(z_e)
                all_latents.append(z_q.cpu())
            elif model_type == 'ortho_vae':
                # OrthoVAE: 使用 encode 方法获取坐标表示
                coordinates = model.encode(x)
                all_latents.append(coordinates.cpu())
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
    
    all_latents = torch.cat(all_latents, dim=0)
    
    # 对隐空间进行标准化处理
    mean = torch.mean(all_latents, dim=(0, 2, 3), keepdim=True)
    std = torch.std(all_latents, dim=(0, 2, 3), keepdim=True)
    
    # 保存标准化参数
    if save_dir is not None:
        torch.save({'mean': mean, 'std': std}, os.path.join(save_dir, 'z_stats.pth'))
        print(f"Saved latent space statistics to {os.path.join(save_dir, 'z_stats.pth')}")
    
    standardized_latents = (all_latents - mean) / std
    
    return standardized_latents, mean, std


def train_diffusion_model(latent_dataset, save_path, args):
    """
    在 SoftVQVAE 的标准化连续隐空间上训练扩散模型.
    """
    # 1. 获取隐空间维度信息
    latent_shape = latent_dataset.shape
    B, C, H, W = latent_shape
    print(f"Latent dataset shape: {latent_shape}. Training U-Net with sample_size={H}, in_channels={C}.")

    # 2. 配置 U-Net 模型
    # 对于8x8的小尺寸输入，需要减少下采样块的数量
    if H <= 8:
        print("Latent space is small. Using a shallow U-Net.")
        # 最多下采样一次
        down_blocks = ("DownBlock2D",)
        up_blocks = ("UpBlock2D",)
        block_channels = (128,) # 减少层数，但可以增加宽度
    else:
        # 默认配置
        down_blocks = ("DownBlock2D", "DownBlock2D")
        up_blocks = ("UpBlock2D", "UpBlock2D")
        block_channels = (64, 128)


    model = UNet2DModel(
        sample_size=H,
        in_channels=C,
        out_channels=C,
        layers_per_block=2,
        block_out_channels=block_channels, 
        down_block_types=down_blocks,
        up_block_types=up_blocks,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. 配置噪声调度器 (Scheduler)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 4. 配置优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.diffusion_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(latent_dataset) // args.batch_size * args.diffusion_epochs),
    )

    # 5. 创建数据加载器
    train_dataset = TensorDataset(latent_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 6. 训练循环
    print(f"--- Training Diffusion Model on Standardized SoftVQVAE latent space ---")
    for epoch in range(args.diffusion_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.diffusion_epochs}")
        
        for step, batch in enumerate(progress_bar):
            clean_latents = batch[0].to(device)
            
            # 采样随机噪声
            noise = torch.randn(clean_latents.shape).to(device)
            
            # 随机采样一个时间步 t
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_latents.shape[0],), device=device).long()
            
            # 根据 t 向干净的隐向量中添加噪声
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
            
            # 预测噪声 (使用正确的forward方法调用)
            noise_pred = model(noisy_latents, timesteps).sample
            
            # 计算损失 (预测的噪声 vs 真实的噪声)
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # 7. 训练完成后, 保存整个 pipeline
    print("Diffusion model training complete. Saving pipeline...")
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.save_pretrained(save_path)
    print(f"Diffusion pipeline saved to {save_path}")
    
    return pipeline


def generate_with_diffusion(diffusion_pipeline, model, model_type, num_samples, device, z_stats_path=None):
    """
    使用扩散模型生成隐向量, 并用相应的模型进行解码.
    支持 ortho_vae 和 softvqvae 两种模型。
    
    Args:
        diffusion_pipeline: 训练好的扩散模型pipeline
        model: 预训练的模型
        model_type: 模型类型，'ortho_vae' 或 'softvqvae'
        num_samples: 生成样本数量
        device: 设备
        z_stats_path: 标准化参数路径
    
    Returns:
        generated_images: 生成的图像
    """
    model.eval()
    
    # 加载标准化参数（如果存在）
    if z_stats_path is not None and os.path.exists(z_stats_path):
        z_stats = torch.load(z_stats_path, map_location=device)
        mean = z_stats['mean'].to(device)
        std = z_stats['std'].to(device)
        print("Loaded latent space statistics for denormalization.")
    else:
        mean = None
        std = None
        print("No latent space statistics found. Using generated latents as-is.")
    
    # 1. 从纯噪声开始, 使用扩散模型 pipeline 生成去噪后的隐向量
    with torch.no_grad():
        output = diffusion_pipeline(
            batch_size=num_samples,
            generator=torch.manual_seed(0), # for reproducibility
            output_type="tensor"
        )
        
        # 确保返回的是PyTorch张量
        if hasattr(output, 'images'):
            generated_latents = output.images
        else:
            generated_latents = output
        
        # 如果返回的是numpy数组，转换为PyTorch张量
        if isinstance(generated_latents, np.ndarray):
            generated_latents = torch.from_numpy(generated_latents)
    
    # 确保张量在正确的设备上
    generated_latents = generated_latents.to(device)
    
    # 检查并调整维度顺序
    if len(generated_latents.shape) == 4:
        # 如果维度是 [batch_size, height, width, channels]，需要转换为 [batch_size, channels, height, width]
        if generated_latents.shape[-1] == 64:  # channels维度在最后
            generated_latents = generated_latents.permute(0, 3, 1, 2)
    
    # 2. 如果存在标准化参数，进行逆标准化
    if mean is not None and std is not None:
        generated_latents = generated_latents * std + mean
        print("Applied denormalization to generated latents.")
    
    # 3. 将生成的潜向量输入解码器得到最终图像
    with torch.no_grad():
        if model_type == 'softvqvae' or model_type == 'ortho_vae':
            generated_images = model.decoder(generated_latents)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    return generated_images


def evaluate_image_quality(real_images, generated_images, device):
    """
    评估生成图像的质量
    """
    # 确保图像在正确的设备上
    real_images = real_images.to(device)
    generated_images = generated_images.to(device)
    
    # 调试信息：检查图像值范围
    print(f"Real images range: [{real_images.min().item():.4f}, {real_images.max().item():.4f}]")
    print(f"Generated images range: [{generated_images.min().item():.4f}, {generated_images.max().item():.4f}]")
    
    # 首先将图像值范围归一化到[0, 1]
    real_images_normalized = (real_images - real_images.min()) / (real_images.max() - real_images.min())
    generated_images_normalized = (generated_images - generated_images.min()) / (generated_images.max() - generated_images.min())
    
    # 然后将图像值范围归一化到[-1, 1]用于LPIPS计算
    real_images_lpips = real_images_normalized * 2 - 1
    generated_images_lpips = generated_images_normalized * 2 - 1
    
    # 检查归一化后的范围
    print(f"Normalized real images range: [{real_images_normalized.min().item():.4f}, {real_images_normalized.max().item():.4f}]")
    print(f"Normalized generated images range: [{generated_images_normalized.min().item():.4f}, {generated_images_normalized.max().item():.4f}]")
    print(f"LPIPS real images range: [{real_images_lpips.min().item():.4f}, {real_images_lpips.max().item():.4f}]")
    print(f"LPIPS generated images range: [{generated_images_lpips.min().item():.4f}, {generated_images_lpips.max().item():.4f}]")
    
    # 初始化评估指标
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    
    # 计算指标
    ssim_score = ssim(generated_images_normalized, real_images_normalized)
    psnr_score = psnr(generated_images_normalized, real_images_normalized)
    lpips_score = lpips(generated_images_lpips, real_images_lpips)
    
    # 计算MSE（使用原始图像）
    mse = F.mse_loss(generated_images, real_images)
    
    return {
        'mse': mse.item(),
        'psnr': psnr_score.item(),
        'ssim': ssim_score.item(),
        'lpips': lpips_score.item()
    }


def get_sample_real_images(data_loader, num_samples, device):
    """
    从数据加载器中获取真实图像样本用于评估
    """
    real_images = []
    for batch_idx, (x, _) in enumerate(data_loader):
        if len(real_images) >= num_samples:
            break
        x = x.to(device)
        real_images.append(x)
    
    # 如果获取的样本不足，重复使用最后一个batch
    if len(real_images) < num_samples:
        remaining = num_samples - len(real_images)
        last_batch = real_images[-1][:remaining]
        real_images.append(last_batch)
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    return real_images


# ==============================================================================
# 3. 主实验流程
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ablation study: PixelCNN vs. Diffusion on VAE Latents')
    
    # 通用参数
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset", type=str, default='CIFAR10')
    parser.add_argument("--vqvae_model_path", type=str, required=True, help="Path to pre-trained VQVAE model.")
    parser.add_argument("--softvqvae_model_path", type=str, default=None, help="Path to pre-trained SoftVQVAE model.")
    parser.add_argument("--ortho_vae_model_path", type=str, default=None, help="Path to pre-trained OrthoVAE model.")
    parser.add_argument("--latent_source", type=str, default='softvqvae', choices=['softvqvae', 'ortho_vae'], 
                        help="Source of latent vectors for diffusion training: 'softvqvae' or 'ortho_vae'.")
    parser.add_argument("--save_dir", type=str, default='ablation_results')
    parser.add_argument("--n_embeddings", type=int, default=512, help="Number of embeddings in VQ codebook.")
    parser.add_argument("--num_samples_to_generate", type=int, default=64, help="Number of images to generate for visualization.")

    # PixelCNN 相关参数
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for PixelCNN training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for PixelCNN.")
    parser.add_argument("--img_dim", type=int, default=64, help="Dimension for PixelCNN.")
    parser.add_argument("--n_layers", type=int, default=15, help="Number of layers in PixelCNN.")

    # Diffusion Model 相关参数
    parser.add_argument("--diffusion_epochs", type=int, default=50, help="Epochs for Diffusion Model training.")
    parser.add_argument("--diffusion_lr", type=float, default=1e-4, help="Learning rate for Diffusion Model.")

    args = parser.parse_args()
    
    # 验证参数
    if args.latent_source == 'softvqvae' and args.softvqvae_model_path is None:
        raise ValueError("When using softvqvae as latent source, --softvqvae_model_path must be provided.")
    elif args.latent_source == 'ortho_vae' and args.ortho_vae_model_path is None:
        raise ValueError("When using ortho_vae as latent source, --ortho_vae_model_path must be provided.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载数据
    training_data, _, training_loader, _, _ = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)

    
     # --- Pipeline 1: VQVAE + PixelCNN ---
    print("\n" + "="*50)
    print("=== Pipeline 1: VQVAE + GatedPixelCNN ===")
    print("="*50)

    # 加载冻结的 VQVAE 模型
    vqvae = VQVAE(h_dim=128, res_h_dim=32, n_res_layers=2, n_embeddings=args.n_embeddings, embedding_dim=64, beta=0.25).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_model_path, map_location=device))
    vqvae.eval()
    print("Loaded frozen VQVAE model.")

    # 1. 创建离散隐空间数据集（包含标签）
    discrete_latents, labels = extract_discrete_latents_vqvae(vqvae, training_loader, device)
    latent_shape_vqvae = discrete_latents.shape[1:] # e.g. (8, 8) or (32, 32)
    print(f"VQVAE discrete latent dataset created. Shape: {discrete_latents.shape}")

    # 2. 训练 PixelCNN 先验模型
    pixelcnn = GatedPixelCNN(input_dim=args.n_embeddings, dim=args.img_dim, n_layers=args.n_layers, n_classes=10).to(device)
    latent_dataset_vqvae = TensorDataset(discrete_latents, labels)
    latent_loader_vqvae = DataLoader(latent_dataset_vqvae, batch_size=args.batch_size, shuffle=True)
    pixelcnn = train_pixelcnn(pixelcnn, "VQVAE", latent_loader_vqvae, None, args, device)
    
    # 保存 PixelCNN 模型
    pixelcnn_save_path = os.path.join(save_dir, "pixelcnn_on_vqvae.pth")
    torch.save(pixelcnn.state_dict(), pixelcnn_save_path)
    print(f"PixelCNN model saved to {pixelcnn_save_path}")

    # 3. 生成新图片
    generated_images_vqvae = generate_with_pixelcnn(pixelcnn, vqvae, args.num_samples_to_generate, latent_shape_vqvae, device)
    save_image(generated_images_vqvae.data.cpu(), os.path.join(save_dir, 'generated_by_pixelcnn.png'), nrow=8, normalize=True)
    print("Generated images from VQVAE+PixelCNN pipeline and saved.")

    # 4. 评估 VQVAE+PixelCNN 生成图像质量
    print("Evaluating VQVAE+PixelCNN image quality...")
    real_images_vqvae = get_sample_real_images(training_loader, args.num_samples_to_generate, device)
    vqvae_quality = evaluate_image_quality(real_images_vqvae, generated_images_vqvae, device)
    
    # 保存评估结果
    with open(os.path.join(save_dir, 'vqvae_pixelcnn_quality.json'), 'w') as f:
        json.dump(vqvae_quality, f, indent=2)
    
    print("VQVAE+PixelCNN Quality Metrics:")
    print(f"  MSE: {vqvae_quality['mse']:.6f}")
    print(f"  PSNR: {vqvae_quality['psnr']:.2f} dB")
    print(f"  SSIM: {vqvae_quality['ssim']:.4f}")
    print(f"  LPIPS: {vqvae_quality['lpips']:.4f}")

    # --- Pipeline 2: VAE + Diffusion Model ---
    print("\n" + "="*50)
    print(f"=== Pipeline 2: {args.latent_source.upper()} + Diffusion Model ===")
    print("="*50)

    # 根据选择的潜向量来源加载相应的模型
    if args.latent_source == 'softvqvae':
        # 加载冻结的 SoftVQVAE 模型
        model = SoftVQVAE(h_dim=128, res_h_dim=32, n_res_layers=2, num_embeddings=args.n_embeddings, embedding_dim=64, beta=0.25,temperature=0.5).to(device)
        model.load_state_dict(torch.load(args.softvqvae_model_path, map_location=device))
        print("Loaded frozen SoftVQVAE model.")
    elif args.latent_source == 'ortho_vae':
        # 加载冻结的 OrthoVAE 模型
        model = OrthoVAE(h_dim=128, res_h_dim=32, n_res_layers=2, num_embeddings=args.n_embeddings, embedding_dim=64, ortho_weight=0.1, entropy_weight=0.1).to(device)
        model.load_state_dict(torch.load(args.ortho_vae_model_path, map_location=device))
        print("Loaded frozen OrthoVAE model.")
    
    model.eval()
    
    # 1. 创建连续隐空间数据集并进行标准化
    continuous_latents, mean, std = extract_continuous_latents(model, args.latent_source, training_loader, device, save_dir)
    print(f"{args.latent_source.upper()} standardized continuous latent dataset created. Shape: {continuous_latents.shape}")
    print(f"Latent space statistics - Mean: {mean.mean().item():.4f}, Std: {std.mean().item():.4f}")

    # 2. 训练 Diffusion Model 先验模型
    diffusion_save_path = os.path.join(save_dir, f"diffusion_pipeline_on_{args.latent_source}")
    diffusion_pipeline = train_diffusion_model(continuous_latents, diffusion_save_path, args)
    
    # 3. 生成新图片（使用标准化参数进行逆标准化）
    z_stats_path = os.path.join(save_dir, 'z_stats.pth')
    generated_images = generate_with_diffusion(diffusion_pipeline, model, args.latent_source, args.num_samples_to_generate, device, z_stats_path)
    save_image(generated_images.data.cpu(), os.path.join(save_dir, f'generated_by_diffusion_{args.latent_source}.png'), nrow=8, normalize=True)
    print(f"Generated images from {args.latent_source.upper()}+Diffusion pipeline and saved.")

    # 4. 评估生成图像质量
    print(f"Evaluating {args.latent_source.upper()}+Diffusion image quality...")
    real_images = get_sample_real_images(training_loader, args.num_samples_to_generate, device)
    quality_metrics = evaluate_image_quality(real_images, generated_images, device)
    
    # 保存评估结果
    with open(os.path.join(save_dir, f'{args.latent_source}_diffusion_quality.json'), 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    
    print(f"{args.latent_source.upper()}+Diffusion Quality Metrics:")
    print(f"  MSE: {quality_metrics['mse']:.6f}")
    print(f"  PSNR: {quality_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {quality_metrics['ssim']:.4f}")
    print(f"  LPIPS: {quality_metrics['lpips']:.4f}")

    # 比较两个pipeline的结果
    print("\n" + "="*50)
    print("=== Pipeline Comparison ===")
    print("="*50)
    print(f"VQVAE+PixelCNN vs {args.latent_source.upper()}+Diffusion:")
    print(f"MSE:     {vqvae_quality['mse']:.6f} vs {quality_metrics['mse']:.6f}")
    print(f"PSNR:    {vqvae_quality['psnr']:.2f} dB vs {quality_metrics['psnr']:.2f} dB")
    print(f"SSIM:    {vqvae_quality['ssim']:.4f} vs {quality_metrics['ssim']:.4f}")
    print(f"LPIPS:   {vqvae_quality['lpips']:.4f} vs {quality_metrics['lpips']:.4f}")

    print("\n" + "="*50)
    print(f"Ablation study complete! All results saved in: {save_dir}")
    print("="*50)


if __name__ == "__main__":
    # 确保你的 VQVAE/SoftVQVAE/OrthoVAE 模型定义和 utils.py 文件在同一个目录下或在 Python 路径中
    # 示例运行命令:
    # python your_script_name.py --vqvae_model_path ./models/vqvae.pth --softvqvae_model_path ./models/soft_vqvae.pth --latent_source softvqvae
    # python your_script_name.py --vqvae_model_path ./models/vqvae.pth --ortho_vae_model_path ./models/ortho_vae.pth --latent_source ortho_vae
    main()