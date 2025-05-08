import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .utils import format_input

class Encoder(nn.Module):
    ''' Encoder architecture
    # - Input -> Block -> MaxPool -> Block -> MaxPool -> Block -> MaxPool -> Block -> MaxPool -> Block -> MaxPool
    # - Output: 1024 channels, HxW reduced by half at each block
    # - Each block reduces HxW by half and changes the number of channels'''
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            Block(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(512, 1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        return self.encoder(x)
    
# Add sigmoid activation to the Decoder class
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            Block(512, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            Block(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            Block(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            Block(64, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Tanh() 
        )
        
    def forward(self, x):
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, beta = 0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.pre_quant_conv = nn.Conv2d(1024, embedding_dim, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.post_quant_conv = nn.Conv2d(embedding_dim, 1024, kernel_size=1)
        
        self.beta = beta
        
    def forward(self, x):
        x = format_input(x)
        # Encoder
        z_e = self.encoder(x)
        z_e = self.pre_quant_conv(z_e)
        
        B, C, H, W = z_e.size()
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape((B, H*W, C))
        
        distances = torch.cdist(z_e_flat, self.embedding.weight[None, :].repeat(B, 1, 1))
        
        # Find nearest embedding
        z_q_indices = torch.argmin(distances, dim=-1)
        z_q = torch.index_select(self.embedding.weight, dim=0, index=z_q_indices.view(-1))
        z_e = z_e_flat.reshape((-1, C))
        
        # losses
        commitment_loss = F.mse_loss(z_q.detach(), z_e) 
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        q_loss = codebook_loss + self.beta * commitment_loss
        
        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2)
        
        latent = z_q
        
        z_q = self.post_quant_conv(z_q)
        x_reconst = self.decoder(z_q)
        
        return x_reconst, latent, q_loss
    
if __name__ == '__main__':
    import torch_directml 
    device = torch_directml.device()
    
    vqvae = VQVAE(128, 512).to(device)
    x = torch.randn(32, 3, 128, 128).to(device)
    x_reconst, latent, q_loss = vqvae(x)
    print(x_reconst.size())
    print(q_loss.size())
    print(q_loss)
    print(q_loss.item())
    print(q_loss.item() / x.size(0))