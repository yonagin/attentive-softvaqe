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


# ==============================================================================
# 1. VQVAE + PixelCNN 部分 (大部分沿用原代码, 稍作调整)
# ==============================================================================

def extract_discrete_latents_vqvae(model, data_loader, device):
    """
    专门为 VQVAE 提取离散的隐空间索引 (latent indices).
    """
    model.eval()
    all_codes = []
    
    with torch.no_grad():
        for x, _ in tqdm(data_loader, desc="Extracting VQVAE Latents"):
            x = x.to(device)
            z = model.encoder(x)
            z = model.pre_quantization_conv(z)
            _, _, _, _, min_encoding_indices = model.vector_quantization(z)
            
            indices = min_encoding_indices.squeeze(1) # [B, 1, H, W] -> [B, H, W]
            all_codes.append(indices.cpu())
    
    return torch.cat(all_codes, dim=0)


def train_pixelcnn(model, model_name, training_loader, validation_loader, args, device):
    """
    在 VQVAE 的离散隐空间上训练 PixelCNN.
    (此函数基本不变)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"--- Training PixelCNN on {model_name} latent space ---")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(training_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for codes in progress_bar:
            codes = codes[0].to(device) # Dataloader returns a list
            
            optimizer.zero_grad()
            logits = model(codes, labels=None) # Unconditional for simplicity
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
# 2. SoftVQVAE + Diffusion Model 部分 (全新实现)
# ==============================================================================

def extract_continuous_latents_softvqvae(model, data_loader, device):
    """
    专门为 SoftVQVAE 提取连续的隐向量 z_q.
    """
    model.eval()
    all_latents = []
    
    with torch.no_grad():
        for x, _ in tqdm(data_loader, desc="Extracting SoftVQVAE Latents"):
            x = x.to(device)
            z = model.encoder(x)
            z = model.pre_quantization_conv(z)
            z_q, _ = model.quantizer(z) # 获取连续的 z_q
            all_latents.append(z_q.cpu())
            
    return torch.cat(all_latents, dim=0)


def train_diffusion_model(latent_dataset, save_path, args):
    """
    在 SoftVQVAE 的连续隐空间上训练扩散模型.
    """
    # 1. 获取隐空间维度信息
    latent_shape = latent_dataset.shape
    B, C, H, W = latent_shape
    print(f"Latent dataset shape: {latent_shape}. Training U-Net with sample_size={H}, in_channels={C}.")

    # 2. 配置 U-Net 模型
    # 这些参数可以根据你的隐空间大小和复杂度进行调整
    model = UNet2DModel(
        sample_size=H,
        in_channels=C,
        out_channels=C,
        layers_per_block=2,
        block_out_channels=(64, 64, 128, 128, 256), # 通道数可以调整
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"
        ),
        up_block_types=(
            "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
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
    print(f"--- Training Diffusion Model on SoftVQVAE latent space ---")
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
            
            # 预测噪声
            noise_pred = model(noisy_latents, timesteps, return_dict=False)[0]
            
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


def generate_with_diffusion(diffusion_pipeline, softvqvae_decoder, num_samples, device):
    """
    使用扩散模型生成隐向量, 并用 SoftVQVAE 解码器解码.
    """
    softvqvae_decoder.eval()
    
    # 1. 从纯噪声开始, 使用扩散模型 pipeline 生成隐向量 z_0
    # The pipeline output is on CPU, we need to move it to the correct device.
    generated_latents = diffusion_pipeline(
        batch_size=num_samples,
        generator=torch.manual_seed(0), # for reproducibility
    ).images
    
    generated_latents = torch.from_numpy(generated_latents).to(device)
    
    # 2. 将生成的隐向量喂给冻结的解码器
    with torch.no_grad():
        generated_images = softvqvae_decoder(generated_latents)
        
    return generated_images


# ==============================================================================
# 3. 主实验流程
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ablation study: PixelCNN vs. Diffusion on VAE Latents')
    
    # 通用参数
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset", type=str, default='CIFAR10')
    parser.add_argument("--vqvae_model_path", type=str, required=True, help="Path to pre-trained VQVAE model.")
    parser.add_argument("--softvqvae_model_path", type=str, required=True, help="Path to pre-trained SoftVQVAE model.")
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载数据
    training_data, _, training_loader, _, _ = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size, num_workers=4)


    # --- Pipeline 2: SoftVQVAE + Diffusion Model ---
    print("\n" + "="*50)
    print("=== Pipeline 2: SoftVQVAE + Diffusion Model ===")
    print("="*50)

    # 加载冻结的 SoftVQVAE 模型
    softvqvae = SoftVQVAE(in_channels=3, h_dim=128, res_h_dim=32, n_res_layers=2, num_embeddings=args.n_embeddings, embedding_dim=64, beta=0.25,temperature=0.5).to(device)
    softvqvae.load_state_dict(torch.load(args.softvqvae_model_path, map_location=device))
    softvqvae.eval()
    print("Loaded frozen SoftVQVAE model.")
    
    # 1. 创建连续隐空间数据集
    continuous_latents = extract_continuous_latents_softvqvae(softvqvae, training_loader, device)
    print(f"SoftVQVAE continuous latent dataset created. Shape: {continuous_latents.shape}")

    # 2. 训练 Diffusion Model 先验模型
    diffusion_save_path = os.path.join(save_dir, "diffusion_pipeline_on_softvqvae")
    diffusion_pipeline = train_diffusion_model(continuous_latents, diffusion_save_path, args)
    
    # 3. 生成新图片
    generated_images_softvqvae = generate_with_diffusion(diffusion_pipeline, softvqvae.decoder, args.num_samples_to_generate, device)
    save_image(generated_images_softvqvae.data.cpu(), os.path.join(save_dir, 'generated_by_diffusion.png'), nrow=8, normalize=True)
    print("Generated images from SoftVQVAE+Diffusion pipeline and saved.")
    
     # --- Pipeline 1: VQVAE + PixelCNN ---
    print("\n" + "="*50)
    print("=== Pipeline 1: VQVAE + GatedPixelCNN ===")
    print("="*50)

    # 加载冻结的 VQVAE 模型
    vqvae = VQVAE(in_channels=3, h_dim=128, res_h_dim=32, n_res_layers=2, n_embeddings=args.n_embeddings, embedding_dim=64, beta=0.25).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_model_path, map_location=device))
    vqvae.eval()
    print("Loaded frozen VQVAE model.")

    # 1. 创建离散隐空间数据集
    discrete_latents = extract_discrete_latents_vqvae(vqvae, training_loader, device)
    latent_shape_vqvae = discrete_latents.shape[1:] # e.g. (8, 8) or (32, 32)
    print(f"VQVAE discrete latent dataset created. Shape: {discrete_latents.shape}")

    # 2. 训练 PixelCNN 先验模型
    pixelcnn = GatedPixelCNN(input_dim=args.n_embeddings, dim=args.img_dim, n_layers=args.n_layers, n_classes=None).to(device)
    latent_dataset_vqvae = TensorDataset(discrete_latents)
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

    print("\n" + "="*50)
    print(f"Ablation study complete! All results saved in: {save_dir}")
    print("="*50)


if __name__ == "__main__":
    # 确保你的 VQVAE/SoftVQVAE 模型定义和 utils.py 文件在同一个目录下或在 Python 路径中
    # 示例运行命令:
    # python your_script_name.py --vqvae_model_path ./models/vqvae.pth --softvqvae_model_path ./models/soft_vqvae.pth
    main()