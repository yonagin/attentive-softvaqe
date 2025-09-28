import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from models.soft_vqvae import SoftVQVAE
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from torchvision import transforms
from scipy import linalg
import lpips
from PIL import Image
import torch.nn.functional as F


def train_model(model, model_name, training_loader, validation_loader, x_train_var, args, device):
    """
    Train a single model and return training results
    """
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    model.train()
    
    results = {
        'model_name': model_name,
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
        'embedding_losses': [],
        'validation_recon_errors': []
    }
    
    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()
        
        # Use the unified forward interface with loss computation
        total_loss, recon_loss, embedding_loss, perplexity, _ = model(x, return_loss=True)
        
        total_loss.backward()
        optimizer.step()
        
        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(total_loss.cpu().detach().numpy())
        results["embedding_losses"].append(embedding_loss.cpu().detach().numpy())
        results["n_updates"] = i
        
        # Validation every 100 updates
        if i % 100 == 0:
            model.eval()
            val_recon_errors = []
            with torch.no_grad():
                for val_batch_idx, (val_x, _) in enumerate(validation_loader):
                    if val_batch_idx >= 10:  # Use first 10 batches for validation
                        break
                    val_x = val_x.to(device)
                    val_recon = model.reconstruct(val_x)
                    val_recon_error = torch.mean((val_recon - val_x)**2) / x_train_var
                    val_recon_errors.append(val_recon_error.cpu().detach().numpy())
            
            results["validation_recon_errors"].append(np.mean(val_recon_errors))
            model.train()
        
        if i % args.log_interval == 0:
            print(f'{model_name} - Update #{i}, '
                  f'Recon Error: {np.mean(results["recon_errors"][-args.log_interval:]):.4f}, '
                  f'Loss: {np.mean(results["loss_vals"][-args.log_interval:]):.4f}, '
                  f'Perplexity: {np.mean(results["perplexities"][-args.log_interval:]):.4f}')
    
    return results


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM between two images
    """
    (_, channel, height, width) = img1.size()
    
    # Create a 1D Gaussian window
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# Global cache for LPIPS model to avoid repeated initialization
_lpips_model_cache = None

def calculate_lpips(model, img1, img2):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity)
    """
    global _lpips_model_cache
    
    # Use LPIPS library if available, otherwise use VGG-based similarity
    try:
        if _lpips_model_cache is None:
            _lpips_model_cache = lpips.LPIPS(net='vgg')
        
        # Move model to the same device as input images
        _lpips_model_cache = _lpips_model_cache.to(img1.device)
        return _lpips_model_cache(img1, img2).mean().item()
    except:
        # Fallback: VGG feature similarity
        vgg = vgg16(pretrained=True).features[:16].eval().to(img1.device)
        
        def normalize_batch(batch):
            # Normalize using ImageNet mean and std
            mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            return (batch - mean) / std
        
        img1_norm = normalize_batch(img1)
        img2_norm = normalize_batch(img2)
        
        with torch.no_grad():
            features1 = vgg(img1_norm)
            features2 = vgg(img2_norm)
            
        # Calculate L2 distance between features
        return F.mse_loss(features1, features2).item()


def calculate_fid(real_imgs, fake_imgs, device):
    """
    Calculate FID (Fréchet Inception Distance)
    Simplified version using VGG features
    """
    vgg = vgg16(pretrained=True).features.eval().to(device)
    
    def extract_features(imgs):
        features = []
        with torch.no_grad():
            for i in range(0, len(imgs), 32):  # Process in batches
                batch = imgs[i:i+32].to(device)  # Move batch to the same device as VGG
                # Normalize for VGG
                mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                batch_norm = (batch - mean) / std
                
                feat = vgg(batch_norm)
                features.append(feat.view(feat.size(0), -1).cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    real_features = extract_features(real_imgs)
    fake_features = extract_features(fake_imgs)
    
    # Calculate FID
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and return comprehensive metrics
    """
    model.eval()
    
    mse_losses = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    # For FID calculation
    all_real_imgs = []
    all_fake_imgs = []
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx >= 100:  # Use more batches for better statistics
                break
                
            x = x.to(device)
            x_recon = model.reconstruct(x)
            
            # Store images for FID
            all_real_imgs.append(x.cpu())
            all_fake_imgs.append(x_recon.cpu())
            
            # MSE
            mse = torch.mean((x_recon - x) ** 2)
            mse_losses.append(mse.cpu().numpy())
            
            # PSNR
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            psnr_values.append(psnr.cpu().numpy())
            
            # SSIM
            ssim = calculate_ssim(x, x_recon)
            ssim_values.append(ssim.cpu().numpy())
            
            # LPIPS
            lpips_val = calculate_lpips(model, x, x_recon)
            lpips_values.append(lpips_val)
    
    # Calculate FID
    real_imgs = torch.cat(all_real_imgs, dim=0)
    fake_imgs = torch.cat(all_fake_imgs, dim=0)
    
    # Limit number of images for FID calculation
    max_fid_samples = min(1000, len(real_imgs))
    fid = calculate_fid(real_imgs[:max_fid_samples], fake_imgs[:max_fid_samples], device)
    
    return {
        'mse': np.mean(mse_losses),
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'lpips': np.mean(lpips_values),
        'fid': fid
    }


def plot_comparison(results_dict, save_dir):
    """
    Plot comparison graphs for different models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training reconstruction error
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for model_name, results in results_dict.items():
        plt.plot(results['recon_errors'], label=model_name)
    plt.xlabel('Training Steps')
    plt.ylabel('Reconstruction Error')
    plt.title('Training Reconstruction Error')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for model_name, results in results_dict.items():
        plt.plot(results['perplexities'], label=model_name)
    plt.xlabel('Training Steps')
    plt.ylabel('Perplexity')
    plt.title('Codebook Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for model_name, results in results_dict.items():
        plt.plot(results['loss_vals'], label=model_name)
    plt.xlabel('Training Steps')
    plt.ylabel('Total Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for model_name, results in results_dict.items():
        if len(results['validation_recon_errors']) > 0:
            val_steps = np.arange(0, len(results['validation_recon_errors'])) * 100
            plt.plot(val_steps, results['validation_recon_errors'], label=model_name)
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Reconstruction Error')
    plt.title('Validation Reconstruction Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Soft-VQVAE vs VQVAE Ablation Study')
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=5000)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset", type=str, default='CIFAR10')
    parser.add_argument("--soft_temperature", type=float, default=1.0)
    
    # Experiment settings
    parser.add_argument("--save_dir", type=str, default='ablation_results')
    parser.add_argument("--save_models", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)
    
    # Create models
    models = {
        'VQVAE': VQVAE(
            args.n_hiddens, args.n_residual_hiddens,
            args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta
        ).to(device),
        'SoftVQVAE': SoftVQVAE(
            args.n_hiddens, args.n_residual_hiddens,
            args.n_residual_layers, args.n_embeddings, args.embedding_dim, 
            args.beta, args.soft_temperature
        ).to(device)
    }
    
    print("Starting ablation experiment...")
    print(f"Models: {list(models.keys())}")
    
    # Train models
    results_dict = {}
    evaluation_results = {}
    
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        results = train_model(
            model, model_name, training_loader, validation_loader, 
            x_train_var, args, device
        )
        
        results_dict[model_name] = results
        
        # Evaluate model
        print(f"Evaluating {model_name}...")
        eval_results = evaluate_model(model, validation_loader, device)
        evaluation_results[model_name] = eval_results
        
        # Save model if requested
        if args.save_models:
            model_path = os.path.join(save_dir, f'{model_name}_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Save results
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in results_dict.items():
            serializable_results[model_name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                for k, v in results.items()
            }
        json.dump(serializable_results, f, indent=2)
    
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Generate comparison plots
    print("Generating comparison plots...")
    plot_comparison(results_dict, save_dir)
    
    # Print final evaluation results
    print("\n=== Final Evaluation Results ===")
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name}:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  PSNR: {results['psnr']:.2f} dB")
        print(f"  SSIM: {results['ssim']:.4f}")
        print(f"  LPIPS: {results['lpips']:.4f}")
        print(f"  FID: {results['fid']:.2f}")
    
    print(f"\nExperiment completed! Results saved to: {save_dir}")


if __name__ == "__main__":
    main()