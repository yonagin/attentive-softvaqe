import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder


class OrthoVAE(nn.Module):
    """
    Orthogonal VAE model with orthogonal basis matrix and entropy regularization.
    """
    
    def __init__(self, h_dim, res_h_dim, n_res_layers, num_embeddings, embedding_dim, ortho_weight=0.1, entropy_weight=0.1):
        super(OrthoVAE, self).__init__()
        # 创建编码器和解码器
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        # 添加预量化卷积层，将编码器输出通道数从h_dim转换为embedding_dim
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 损失函数权重
        self.ortho_weight = ortho_weight
        self.entropy_weight = entropy_weight
        
        # 码本现在是一个可学习的正交基矩阵, 使用 nn.Parameter 更纯粹
        self.basis_matrix = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        # 使用正交初始化, 帮助模型更快进入状态
        nn.init.orthogonal_(self.basis_matrix)

    def forward(self, x, return_loss=False, noise_std=0.0):
        """
        Forward pass with optional loss computation and noise injection
        
        Args:
            x: input tensor
            return_loss: whether to return loss components
            noise_std: standard deviation of Gaussian noise to add to latent vectors
        """
        # 1. 编码
        ze = self.encoder(x)
        ze = self.pre_quantization_conv(ze)
        
        # 添加噪声到隐向量
        if noise_std > 0.0:
            ze = ze + torch.randn_like(ze) * noise_std
        
        # 2. 投影得到坐标
        b, c, h, w = ze.shape
        ze_flat = ze.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        coordinates_flat = torch.matmul(ze_flat, self.basis_matrix.t())
        
        # 3. 解码
        coordinates = coordinates_flat.view(b, h, w, self.num_embeddings).permute(0, 3, 1, 2).contiguous()
        x_recon = self.decoder(coordinates)
        
        if return_loss:
            # 计算损失
            # --- 损失1: 重建损失 (核心) ---
            recon_loss = F.mse_loss(x_recon, x)
            
            # --- 损失2: 基矩阵正交损失 ---
            C = self.basis_matrix
            product = torch.matmul(C, C.t())
            identity = torch.eye(self.num_embeddings, device=C.device)
            ortho_loss = torch.mean((product - identity).pow(2))
            
            # --- 损失3: 坐标使用率熵损失 ---
            probs_flat = F.softmax(coordinates_flat, dim=-1)
            avg_probs = torch.mean(probs_flat, dim=0)
            
            max_entropy = torch.log(torch.tensor(self.num_embeddings, device=avg_probs.device))
            entropy_loss = max_entropy + (avg_probs * torch.log(avg_probs + 1e-9)).sum()
            
            # --- 最终总损失 ---
            total_loss = recon_loss + self.ortho_weight * ortho_loss + self.entropy_weight * entropy_loss
            
            return total_loss, recon_loss, ortho_loss, entropy_loss, x_recon
        else:
            return x_recon
    
    @torch.no_grad()
    def reconstruct(self, x):
        """
        重建输入，用于推理阶段
        """
        return self.forward(x, return_loss=False)
    
    @torch.no_grad()
    def encode(self, x):
        """
        编码输入图像为坐标表示
        """
        # 1. 编码
        ze = self.encoder(x)
        ze = self.pre_quantization_conv(ze)
        
        # 2. 投影得到坐标
        b, c, h, w = ze.shape
        ze_flat = ze.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        coordinates_flat = torch.matmul(ze_flat, self.basis_matrix.t())
        
        # 3. 转换为坐标张量
        coordinates = coordinates_flat.view(b, h, w, self.num_embeddings).permute(0, 3, 1, 2).contiguous()
        return coordinates