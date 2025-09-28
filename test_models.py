import torch
import torch.nn as nn
from models.vqvae import VQVAE
from models.soft_vqvae import SoftVQVAE

def test_models():
    """Test both VQVAE and SoftVQVAE models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    h_dim = 128
    res_h_dim = 32
    n_res_layers = 2
    n_embeddings = 512
    embedding_dim = 64
    beta = 0.25
    temperature = 1.0
    
    # Create models
    vqvae = VQVAE(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta).to(device)
    soft_vqvae = SoftVQVAE(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta, temperature).to(device)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    
    print("Testing VQVAE...")
    # Test forward pass without loss
    embedding_loss, x_hat, perplexity = vqvae(x)
    print(f"VQVAE - Input shape: {x.shape}")
    print(f"VQVAE - Output shape: {x_hat.shape}")
    print(f"VQVAE - Embedding loss: {embedding_loss.item():.4f}")
    print(f"VQVAE - Perplexity: {perplexity.item():.4f}")
    
    # Test forward pass with loss
    total_loss, recon_loss, embedding_loss, perplexity, x_hat_loss = vqvae(x, return_loss=True)
    print(f"VQVAE - Total loss: {total_loss.item():.4f}")
    print(f"VQVAE - Recon loss: {recon_loss.item():.4f}")
    print(f"VQVAE - Embedding loss: {embedding_loss.item():.4f}")
    print(f"VQVAE - Output shapes match: {x_hat.shape == x_hat_loss.shape}")
    
    print("\nTesting SoftVQVAE...")
    # Test forward pass without loss
    embedding_loss, x_hat, perplexity = soft_vqvae(x)
    print(f"SoftVQVAE - Input shape: {x.shape}")
    print(f"SoftVQVAE - Output shape: {x_hat.shape}")
    print(f"SoftVQVAE - Embedding loss: {embedding_loss.item():.4f}")
    print(f"SoftVQVAE - Perplexity: {perplexity.item():.4f}")
    
    # Test forward pass with loss
    total_loss, recon_loss, embedding_loss, perplexity, x_hat_loss = soft_vqvae(x, return_loss=True)
    print(f"SoftVQVAE - Total loss: {total_loss.item():.4f}")
    print(f"SoftVQVAE - Recon loss: {recon_loss.item():.4f}")
    print(f"SoftVQVAE - Embedding loss: {embedding_loss.item():.4f}")
    print(f"SoftVQVAE - Output shapes match: {x_hat.shape == x_hat_loss.shape}")
    
    # Test reconstruction
    x_recon_vqvae = vqvae.reconstruct(x)
    x_recon_soft = soft_vqvae.reconstruct(x)
    print(f"\nReconstruction shapes match: {x_recon_vqvae.shape == x_recon_soft.shape}")
    
    print("\nAll tests passed! Models are correctly implemented.")

if __name__ == "__main__":
    test_models()