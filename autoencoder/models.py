'''# The Encoder and Decoder follows a U-net like architecture
# The Encoder downsamples the input image to a latent space of size 1024x8x8
# The Decoder upsamples the latent space back to the original image size'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Select GPU device if available, otherwise use CPU
import torch_directml
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_directml.is_available():
    device = torch_directml.device()
else:
    device = torch.device('cpu')

def format_input(x):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(x.shape) != 4:
        raise ValueError(f"Input tensor must be 4D - B, C, H, W. Got {x.shape}")
    return x


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
            nn.Sigmoid()  # Add sigmoid activation for [0,1] pixel values
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
        
        self.fc_mu = nn.Linear(1024*4*4, latent_dim)
        self.fc_var = nn.Linear(1024*4*4, latent_dim)
        self.fc = nn.Linear(latent_dim, 1024*4*4)
        
        self.beta = beta
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        x = format_input(x)
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1) # Flatten the tensor
        mu, logvar = self.fc_mu(x_flat), self.fc_var(x_flat)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = format_input(x)
        z, mu, logvar = self.encode(x)
        x_reconst = self.decode(z)
        return x_reconst, mu, logvar
    
    def loss_function(self, x, x_reconst, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconst, x, reduction='sum') / x.size(0)
        # KL divergence loss
        kl_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta*kl_loss
    
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
        z_q = torch.index_select(self.embedding.weight, 0, z_q_indices.view(-1))
        z_e = z_e_flat.reshape((-1, C))
        
        # losses
        commitment_loss = F.mse_loss(z_q.detach(), z_e) 
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        q_loss = codebook_loss + self.beta * commitment_loss
        
        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2)
        
        z_q = self.post_quant_conv(z_q)
        x_reconst = self.decoder(z_q)
        
        return x_reconst, q_loss
        
    

if __name__ == '__main__':
    vae = VAE(128).to(device)
    x = torch.randn(32, 3, 128, 128).to(device)
    z, mu, logvar = vae.encode(x)
    print(z.size())
    print(mu.size())
    print(logvar.size())
    x_reconst = vae.decode(z)
    print(x_reconst.size())
    
    vqvae = VQVAE(128, 512).to(device)
    x = torch.randn(32, 3, 128, 128).to(device)
    x_reconst, q_loss = vqvae(x)
    print(x_reconst.size())
    print(q_loss.size())
    print(q_loss)
    print(q_loss.item())
    print(q_loss.item() / x.size(0))