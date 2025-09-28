import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.encoder import Encoder
from models.quantizer import SoftVQ
from models.decoder import Decoder


class SoftVQVAE(nn.Module):
    """
    Soft VQ-VAE model with soft quantization.
    """
    
    def __init__(self, h_dim, res_h_dim, n_res_layers, num_embeddings, embedding_dim, beta=0.25, temperature=1.0):
        super(SoftVQVAE, self).__init__()
        # 创建编码器和解码器
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        # 添加预量化卷积层，将编码器输出通道数从h_dim转换为embedding_dim
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        self.quantizer = SoftVQ(num_embeddings, embedding_dim, temperature)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.beta = beta

    def loss(self, x):
        ze = self.encoder(x)
        ze = self.pre_quantization_conv(ze)
        zq, _ = self.quantizer(ze)
        # 使用ze.detach()来阻止梯度流向量化编码器
        codebook_loss = F.mse_loss(ze.detach(), zq)
        
        x_recon = self.decoder(zq)
        
        recon_loss = F.mse_loss(x_recon, x)
        
        total_loss = recon_loss + self.beta * codebook_loss
        
        return (total_loss, recon_loss, codebook_loss)
    
    def forward(self, x):
        """
        Forward pass
        """
        ze = self.encoder(x)
        ze = self.pre_quantization_conv(ze)
        zq, _ = self.quantizer(ze)
        x_hat = self.decoder(zq)
        return x_hat
    
    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct input
        """
        ze = self.encoder(x)
        ze = self.pre_quantization_conv(ze)
        zq, _ = self.quantizer(ze)
        x_hat = self.decoder(zq)
        return x_hat


if __name__ == "__main__":
    # Test the model
    encoder = Encoder(3, 128, 2, 32)
    decoder = Decoder(64, 128, 2, 32)
    model = SoftVQVAE(
        encoder=encoder,
        decoder=decoder,
        num_embeddings=512,
        embedding_dim=64,
        beta=0.25,
        temperature=1.0
    )
    
    # Random input
    x = torch.randn(2, 3, 32, 32)
    x_hat = model(x)
    total_loss, recon_loss, codebook_loss = model.loss(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"Codebook loss: {codebook_loss.item():.4f}")