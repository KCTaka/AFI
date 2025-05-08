import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import format_input # Use relative import for utils as well, assuming it's in the same directory

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Find embed_dim and number of heads
        self.embed_dim = out_channels
        self.num_heads = 4
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.sa = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
        
        # Skip connection for channel dimension matching
        self.skip_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        init_x = x
        x = self.block1(x)
        x = self.block2(x)
        
        # Apply skip connection with dimension matching
        if self.skip_conv1 is not None:
            init_x = self.skip_conv1(init_x)
            
        return x + init_x
    
class SelfAttentionBlock(nn.Module): 
    def __init__(self, in_channels, num_heads=4):
        super(SelfAttentionBlock, self).__init__()
        self.embed_dim = in_channels
        self.num_heads = num_heads
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.sa = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.reshape(B, C, H * W)
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # (B, H*W, C)
        x = self.sa(x, x, x)[0]
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H*W) -> (B, C, H, W) 
        return x
    
class SelfAttentionWithResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttentionWithResnetBlock, self).__init__()
        self.resnet_block = ResNetBlock(in_channels, out_channels)
        self.self_attention_block = SelfAttentionBlock(out_channels)
        
    def forward(self, x):
        x1 = self.resnet_block(x)
        x = self.self_attention_block(x1)
        return x + x1

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
        
        # self.encoder = nn.Sequential(
        #     Block(3, 64),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Block(64, 128),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Block(128, 256),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Block(256, 512),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Block(512, 1024),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
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
        
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        #     Block(512, 512),
        #     nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        #     Block(256, 256),
        #     nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        #     Block(128, 128),
        #     nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        #     Block(64, 64),
        #     nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
        #     nn.Tanh()
        # )
        
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

