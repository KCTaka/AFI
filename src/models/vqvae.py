import autoroot
import autorootcwd

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import DownBlock, MidBlock, UpBlock
from src.utils.formats import format_input



class Encoder(nn.Module):
    def __init__(self, in_channels, 
                 latent_dim = 4,
                 down_channels = [64, 128, 256, 256],
                 mid_channels = [256, 256],
                 downsamples = [True, True, True], 
                 down_attn = [False, False, False],
                 num_heads = 4,
                 num_down_layers = 2,
                 num_mid_layers = 2):
        
        super(Encoder, self).__init__()
        
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down_blocks.append(
                DownBlock(
                    down_channels[i], down_channels[i+1],
                    downsample=downsamples[i], num_heads=num_heads,
                    num_layers=num_down_layers, use_self_attention=down_attn[i]
                )
            )

        self.mid_blocks = nn.ModuleList()
        for i in range(len(mid_channels)-1):
            self.mid_blocks.append(
                MidBlock(
                    down_channels[-1], out_channels=mid_channels[i],
                    num_heads=num_heads, num_layers=num_mid_layers
                )
            )
            
        self.out_block = nn.Sequential(
            nn.BatchNorm2d(mid_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(mid_channels[-1], latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = format_input(x)
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        for block in self.mid_blocks:
            x = block(x)
        x = self.out_block(x)
        return x


# Add sigmoid activation to the Decoder class
class Decoder(nn.Module):
    def __init__(self, im_channels, 
                 latent_dim = 4,
                 up_channels = [256, 256, 128, 64],
                 mid_channels = [256, 256],
                 upsamples = [True, True, True],
                 up_attn = [False, False, False],
                 num_heads = 4,
                 num_up_layers = 2,
                 num_mid_layers = 2):

        super(Decoder, self).__init__()
        
        self.conv_in = nn.Conv2d(latent_dim, up_channels[0], kernel_size=3, padding=1)
        
        self.mid_blocks = nn.ModuleList()
        for i in range(len(mid_channels)-1):
            self.mid_blocks.append(
                MidBlock(
                    up_channels[i], out_channels=mid_channels[i],
                    num_heads=num_heads, num_layers=num_mid_layers
                )
            )
        
        self.up_blocks = nn.ModuleList()
        for i in range(len(up_channels)-1):
            self.up_blocks.append(
                UpBlock(
                    up_channels[i], up_channels[i+1],
                    upsample=upsamples[i], num_heads=num_heads,
                    num_layers=num_up_layers, use_self_attention=up_attn[i]
                )
            )
        
        self.out_model = nn.Sequential(
            nn.BatchNorm2d(up_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(up_channels[-1], im_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        x = format_input(x)
        x = self.conv_in(x)
        for block in self.mid_blocks:
            x = block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.out_model(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, 
                    embedding_dim = 4, 
                    num_embeddings = 8192, 
                    beta=0.25,
                    im_channels = 3,
                    down_channels = [64, 128, 256, 256],
                    mid_channels = [256, 256],
                    downsamples = [True, True, True],
                    down_attn = [False, False, False],
                    num_heads = 4,
                    num_down_layers = 2,
                    num_mid_layers = 2,
                    num_up_layers = 2,
                    ):
        
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels=im_channels,
                                latent_dim=embedding_dim,
                                down_channels=down_channels,
                                mid_channels=mid_channels,
                                downsamples=downsamples,
                                down_attn=down_attn,
                                num_heads=num_heads,
                                num_down_layers=num_down_layers,
                                num_mid_layers=num_mid_layers)
        
        self.decoder = Decoder(im_channels=im_channels,
                                latent_dim=embedding_dim,
                                up_channels=down_channels[::-1],
                                mid_channels=mid_channels[::-1],
                                upsamples=downsamples[::-1],
                                up_attn=down_attn[::-1],
                                num_heads=num_heads,
                                num_up_layers=num_up_layers)

        self.pre_quant_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.post_quant_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        
        self.beta = beta
        
    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quant_conv(z_e)
        B, C, H, W = z_e.size()
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape((B, H*W, C))
        distances = torch.cdist(z_e_flat, self.embedding.weight[None, :].repeat(B, 1, 1))
        
        # Find nearest embedding
        z_q_indices = torch.argmin(distances, dim=-1)
        z_q = torch.index_select(self.embedding.weight, dim=0, index=z_q_indices.view(-1))
        
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return z_q
    
    def decode(self, z_q):
        z_q = self.post_quant_conv(z_q)
        x_reconst = self.decoder(z_q)
        return x_reconst
        
    def forward(self, x):
        x = format_input(x) # (B, C, H, W)
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
    
    from matplotlib import pyplot as plt
    from src.utils.visualizers import convert_to_target_visible_channels
    
    vqvae = VQVAE(
        embedding_dim=4,
        num_embeddings=8192,
        beta=0.25,
        im_channels=3,
        down_channels=[64, 128, 256, 256],
        mid_channels=[256, 256],
        downsamples=[True, True, True],
        down_attn=[False, False, False],
        num_heads=4,
        num_down_layers=2,
        num_mid_layers=2,
        num_up_layers=2,
    ).to(device)
    
    x = torch.randn(8, 3, 128, 128).to(device)
    
    vqvae.eval()                # freeze BatchNorm in both encoder & decoder

    x_reconst, latent, q_loss = vqvae(x)
    
    # round‐trip encode/decode
    z_q = vqvae.encode(x)
    x_reconst_from_encode = vqvae.decode(z_q)
    
    assert torch.allclose(x_reconst, x_reconst_from_encode, atol=1e-5), (
        "Reconstructed image from encode-decode does not match original reconstruction."
    )
    assert torch.allclose(latent, z_q, atol=1e-5), (
        "Latent representation from encode does not match the latent "
        "representation from the forward pass."
    )
    
    print("Passed .encode-decode consistency check.")
    
    # Visualize the original image, latent representation, and reconstructed image
    # latent_visible_image = convert_to_target_visible_channels(latent, target_channels=3)[0] # C, H, W
    # plt.imshow(latent_visible_image.permute(1, 2, 0).detach().cpu().numpy())
    # plt.axis('off')
    # plt.show()
    
    # x_reconst = convert_to_target_visible_channels(x_reconst, target_channels=3)[0] # C, H, W
    # plt.imshow(x_reconst.permute(1, 2, 0).detach().cpu().numpy())
    # plt.axis('off')
    # plt.show()

    # print(x_reconst.size())
    # print(latent.size())
    # print(q_loss.size())
    # print(q_loss)
    # print(q_loss.item())
    # print(q_loss.item() / x.size(0))
    # print(vqvae)
    