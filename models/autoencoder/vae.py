import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block # Use relative import
from .utils import format_input # Use relative import for utils as well, assuming it's in the same directory

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

# Explain the VAE architecture
class VAE(nn.Module):
    def __init__(self, latent_dim, beta = 1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.fc_encoder = nn.Linear(1024*4*4, latent_dim*2)
        self.fc_decoder = nn.Linear(latent_dim, 1024*4*4)
        
        self.beta = beta
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        x = format_input(x)
        x = self.encoder(x)
        print("Encoder output size:", x.size())
        x_flat = x.view(x.size(0), -1) # Flatten the tensor
        mu, logvar = self.fc_encoder(x_flat).chunk(2, dim=-1) # Split into mu and logvar
        print("Mu size:", mu.size(), "Logvar size:", logvar.size())
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.decoder(x)
        print("Decoded output size:", x.size())
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

