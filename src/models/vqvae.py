import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.helpers import format_input
from src.models.blocks import ResNetBlock, SelfAttentionWithResnetBlock

class Block(nn.Module):
    '''    # Block architecture
    # - Convolutional layers with ReLU activation and Batch Normalization
    # - Input -> Convolution -> ReLU -> BatchNorm -> Convolution -> ReLU -> BatchNorm -> Output
    # - Each block reduces HxW by half and changes the number of channels'''
    
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Skip connection for channel dimension matching
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
    
    def forward(self, x):
        init_x = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # Apply skip connection with dimension matching
        if self.skip is not None:
            init_x = self.skip(init_x)
            
        x = F.relu(x + init_x) # Residual connection
        
        return x


class Encoder(nn.Module):
    ''' Encoder architecture
    # - Input -> Block -> MaxPool -> Block -> MaxPool -> Block -> MaxPool -> Block -> MaxPool -> Block -> MaxPool
    # - Output: 1024 channels, HxW reduced by half at each block
    # - Each block reduces HxW by half and changes the number of channels'''
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        self.block1 = nn.Sequential(
            ResNetBlock(32, 64),
            ResNetBlock(64, 64),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block1_1 = nn.Sequential(
            ResNetBlock(64, 128),
            ResNetBlock(128, 128),
        ) 
        
        self.block1_2 = nn.Sequential(
            ResNetBlock(128, 256),
            ResNetBlock(256, 256),
        )
        
        self.resnet1 = ResNetBlock(256, 512)
        self.block2 = nn.Sequential(
            SelfAttentionWithResnetBlock(512, 1024),
            SelfAttentionWithResnetBlock(1024, 1024),
        )
        
        self.block3 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            nn.Conv2d(1024, latent_dim, kernel_size=3, padding=1),  
        )
        
        self.encoder = nn.Sequential(
            self.conv1,
            self.block1,
            self.maxpool,
            self.block1_1,
            self.maxpool,
            self.block1_2,
            self.maxpool,
            self.resnet1,
            self.block2,
            self.block3,
        )
        
    def forward(self, x):
        return self.encoder(x)
    
# Add sigmoid activation to the Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(latent_dim, 1024, kernel_size=3, padding=1)
        
        self.resnet1 = ResNetBlock(1024, 512)
        self.block1 = nn.Sequential(
            SelfAttentionWithResnetBlock(512, 256),
            SelfAttentionWithResnetBlock(256, 256),
        )
        
        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            ResNetBlock(256, 128),
            ResNetBlock(128, 128),
        )
        
        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.block2_1 = nn.Sequential(
            ResNetBlock(128, 64),
            ResNetBlock(64, 64),
        )
        
        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.block2_2 = nn.Sequential(
            ResNetBlock(64, 32),
            ResNetBlock(32, 32),
        )
        
        self.block3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
        )
        
        self.decoder = nn.Sequential(
            self.conv1,   
            self.resnet1,
            self.block1,
            self.upconv1,
            self.block2,
            self.upconv2,
            self.block2_1,
            self.upconv3,
            self.block2_2,
            self.block3,
            nn.Tanh(),
        )
        
    def forward(self, x):
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, beta = 0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        
        self.pre_quant_conv = nn.Identity() #nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.post_quant_conv = nn.Identity() #nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        
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