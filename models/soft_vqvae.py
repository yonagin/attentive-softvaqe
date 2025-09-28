import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.encoder import Encoder
from models.quantizer import SoftVectorQuantizer
from models.decoder import Decoder


class SoftVQVAE(nn.Module):
    """
    Soft VQ-VAE model with soft quantization.
    """
    
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, temperature=1.0, save_img_embedding_map=False):
        super(SoftVQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through soft discretization bottleneck
        self.vector_quantization = SoftVectorQuantizer(
            n_embeddings, embedding_dim, beta, temperature)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False, return_loss=False):
        """
        Forward pass with optional loss computation
        
        Args:
            x: input tensor
            verbose: whether to print shapes
            return_loss: whether to return loss components
        """
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        if return_loss:
            recon_loss = F.mse_loss(x_hat, x)
            total_loss = recon_loss + embedding_loss
            return total_loss, recon_loss, embedding_loss, perplexity, x_hat
        else:
            return embedding_loss, x_hat, perplexity
    
    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct input
        """
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        _, z_q, _, _, _ = self.vector_quantization(z_e)
        x_recon = self.decoder(z_q)
        return x_recon


if __name__ == "__main__":
    # Test the model
    model = SoftVQVAE(
        h_dim=128,
        res_h_dim=32,
        n_res_layers=2,
        n_embeddings=512,
        embedding_dim=64,
        beta=0.25,
        temperature=1.0
    )
    
    # Random input
    x = torch.randn(2, 3, 32, 32)
    embedding_loss, x_hat, perplexity = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"Embedding loss: {embedding_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")