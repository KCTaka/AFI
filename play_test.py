import torch
import torch.nn as nn
import torch_directml 

from models.autoencoder.vqvae import VQVAE
from models.autoencoder.vae import VAE, ResNetBlock
from models.autoencoder.discriminator import Discriminator


def test_vae():
    device = torch_directml.device()
    vae = VAE(128).to(device)
    x = torch.randn(8, 3, 128, 128).to(device)
    # z, mu, logvar = vae.encode(x)
    # x_reconst = vae.decode(z)

    # num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    # print("Number of parameters in VAE:", num_params)

    # print("latent vector size:", z.size())
    # print("mu size:", mu.size())
    # print("logvar size:", logvar.size())

    # print("reconstructed image size:", x_reconst.size())
    
    forward_pass, latent_vector, kld_loss = vae(x)
    print("Forward pass output size:", forward_pass.size())
    print("Latent vector size:", latent_vector.size())
    print("KL divergence loss:", kld_loss.item())
    
    
    print("Testing VAE completed.")
    
    
def test_vqvae():
    device = torch_directml.device()
    vqvae = VQVAE(embedding_dim=128, num_embeddings=512).to(device)
    x = torch.randn(8, 3, 128, 128).to(device)
    x_reconst, z, q_loss = vqvae(x)

    print("Reconstructed image size:", x_reconst.size())
    print("Quantization loss:", q_loss.item())

    print("Testing VQVAE completed.")
    
def test_discriminator():
    device = torch_directml.device()
    discriminator = Discriminator(in_channels=3).to(device)
    x = torch.randn(32, 3, 128, 128).to(device)
    output = discriminator(x)

    print("Discriminator output size:", output.size())

    print("Testing Discriminator completed.")

def test_sa():
    device = torch_directml.device()

    # Parameters
    embed_dim = 64  # Embedding dnsion
    num_heads = 8   # Number of attention heads

    # Initialize MultiheadAttention
    self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    # Sample input
    batch_size = 32
    seq_length = 10
    x = torch.rand(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)
    
    print("Input size:", x.size())

    # Since it's self-attention, query, key, and value are all x
    output, attn_weights = self_attn(x, x, x)
    
    print("Self-attention output size:", output.size())
    print("Attention weights size:", attn_weights.size())
    

    assert output.size() == (batch_size, seq_length, embed_dim), "Output size mismatch"
    assert attn_weights.size() == (batch_size, num_heads, seq_length, seq_length), "Attention weights size mismatch"
    
def test_resnet_block():
    device = torch_directml.device()
    block = ResNetBlock(in_channels=16, out_channels=64).to(device)
    x = torch.randn(32, 16, 32, 32).to(device)
    output = block(x)

    print("ResNet Block output size:", output.size())

    print("Testing ResNet Block completed.")
    
        
if __name__ == '__main__':
    test_discriminator()