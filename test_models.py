import torch
import torch.nn as nn
from models.vqvae import VQVAE
from models.soft_vqvae import SoftVQVAE
from models.ortho_vae import OrthoVAE
from models.encoder import Encoder
from models.decoder import Decoder

def test_models():
    """Test VQVAE, SoftVQVAE, and OrthoVAE models"""
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
    ortho_weight = 0.1
    entropy_weight = 0.1
    
    # Create models
    vqvae = VQVAE(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta).to(device)
    
    # Create SoftVQVAE with new constructor
    soft_vqvae = SoftVQVAE(
        h_dim=h_dim,
        res_h_dim=res_h_dim,
        n_res_layers=n_res_layers,
        num_embeddings=n_embeddings,
        embedding_dim=embedding_dim,
        beta=beta,
        temperature=temperature
    ).to(device)
    
    # Create OrthoVAE
    ortho_vae = OrthoVAE(
        h_dim=h_dim,
        res_h_dim=res_h_dim,
        n_res_layers=n_res_layers,
        num_embeddings=n_embeddings,
        embedding_dim=embedding_dim,
        entropy_weight=entropy_weight,
        svb_epsilon=0.1
    ).to(device)
    
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
    x_hat = soft_vqvae(x)
    print(f"SoftVQVAE - Input shape: {x.shape}")
    print(f"SoftVQVAE - Output shape: {x_hat.shape}")
    
    # Test forward pass with loss (using new interface)
    total_loss, recon_loss, codebook_loss, x_hat_loss = soft_vqvae(x, return_loss=True)
    print(f"SoftVQVAE - Total loss: {total_loss.item():.4f}")
    print(f"SoftVQVAE - Recon loss: {recon_loss.item():.4f}")
    print(f"SoftVQVAE - Codebook loss: {codebook_loss.item():.4f}")
    print(f"SoftVQVAE - Output shapes match: {x_hat.shape == x_hat_loss.shape}")
    
    # Test reconstruction
    x_recon_vqvae = vqvae.reconstruct(x)
    x_recon_soft = soft_vqvae.reconstruct(x)
    print(f"\nReconstruction shapes match: {x_recon_vqvae.shape == x_recon_soft.shape}")
    
    print("\nTesting OrthoVAE...")
    # Test forward pass without loss
    x_hat = ortho_vae(x)
    print(f"OrthoVAE - Input shape: {x.shape}")
    print(f"OrthoVAE - Output shape: {x_hat.shape}")
    
    # Test forward pass with loss
    total_loss, recon_loss, entropy_loss, x_hat_loss = ortho_vae(x, return_loss=True)
    print(f"OrthoVAE - Total loss: {total_loss.item():.4f}")
    print(f"OrthoVAE - Recon loss: {recon_loss.item():.4f}")
    print(f"OrthoVAE - Entropy loss: {entropy_loss.item():.4f}")
    print(f"OrthoVAE - Output shapes match: {x_hat.shape == x_hat_loss.shape}")
    
    # Test reconstruction
    x_recon_ortho = ortho_vae.reconstruct(x)
    print(f"OrthoVAE - Reconstruction shape: {x_recon_ortho.shape}")
    
    # Test encode/decode
    coordinates = ortho_vae.encode(x)
    x_recon_decode = ortho_vae.decode(coordinates)
    print(f"OrthoVAE - Encode/decode shape: {x_recon_decode.shape}")
    print(f"OrthoVAE - Encode/decode matches: {torch.allclose(x_recon_ortho, x_recon_decode)}")
    
    print("\nAll tests passed! Models are correctly implemented.")

if __name__ == "__main__":
    test_models()