import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.helpers import format_input
from src.models.blocks import ResNetBlock, SelfAttentionWithResnetBlock

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
            nn.Conv2d(1024, 2 * latent_dim, kernel_size=3, padding=1),  
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

# Explain the VAE architecture
class VAE(nn.Module):
    def __init__(self, latent_dim, beta = 1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.beta = beta
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        x = format_input(x)
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1) # Split into mu and logvar
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        x = self.decoder(z)
        return x
    
    def forward(self, x): # Ensure it outs a list/tuple or multiple outputs
        x = format_input(x)
        z, mu, logvar = self.encode(x)
        x_reconst = self.decode(z)
        kld_loss = self.kldloss(x, x_reconst, mu, logvar)
        return x_reconst, z, self.beta*kld_loss
    
    def kldloss(self, x, x_reconst, mu, logvar):
        # KL divergence loss
        kld_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return kld_loss
    
    
if __name__ == '__main__':
    import torch_directml 
    device = torch_directml.device()
    vae = VAE(128).to(device)
    x = torch.randn(32, 3, 128, 128).to(device)
    z, mu, logvar = vae.encode(x)
    x_reconst = vae.decode(z)
    
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print("Number of parameters in VAE:", num_params)
    
    print("latent vector size:", z.size())
    print("mu size:", mu.size())
    print("logvar size:", logvar.size())
    
    print("reconstructed image size:", x_reconst.size())

