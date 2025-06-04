import autoroot
import autorootcwd

import torch
import torch.nn as nn

from src.models.blocks import DownBlock, MidBlock, UpBlock
from src.utils.embedding import get_time_embedding

class UNet(nn.Module):
    def __init__(self,
                 in_channels = 4,
                 out_channels = 4,
                 down_channels = [64, 128, 256, 256],
                 mid_channels = [256, 256],
                 t_emb_dim = 512,
                 downsamples = [True, True, True],
                 num_down_layers = 2,
                 num_mid_layers = 2,
                 num_up_layers = 2,
                 self_attn = [True, True, True],
                 num_heads = 4,
                 conv_out_channels = 128
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.t_emb_dim = t_emb_dim
        self.downsamples = downsamples
        self.num_down_layers = num_down_layers
        self.num_mid_layers = num_mid_layers
        self.num_up_layers = num_up_layers
        self.self_attn = self_attn
        self.num_heads = num_heads
        
        self.time_emb_layer = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down_blocks.append(
                DownBlock(
                    down_channels[i], down_channels[i+1],
                    t_emb_dim=t_emb_dim,
                    downsample=downsamples[i], num_heads=num_heads,
                    num_layers=num_down_layers,
                    use_self_attention=self.self_attn[i]
                )
            )
        
        self.mid_blocks = nn.ModuleList()
        for i in range(len(mid_channels)-1):
            self.mid_blocks.append(
                MidBlock(
                    mid_channels[i], mid_channels[i+1],
                    t_emb_dim=t_emb_dim,
                    num_heads=num_heads, num_layers=num_mid_layers
                )
            )
        
        self.up_channels = down_channels[-2::-1] + [conv_out_channels]
        self.upsamples = downsamples[::-1]
        self.up_attn = self.self_attn[::-1]
        self.up_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.up_blocks.append(
                UpBlock(
                    self.up_channels[i]*2, self.up_channels[i+1],
                    t_emb_dim=t_emb_dim,
                    upsample=self.upsamples[i],
                    num_heads=num_heads,
                    num_layers=num_up_layers,
                    use_self_attention=self.up_attn[i],
                    require_down_cat=True,
                )
            )
        
        self.out_block = nn.Sequential(
            nn.BatchNorm2d(self.up_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(self.up_channels[-1], self.out_channels, kernel_size=3, padding=1),
        )
            
    def forward(self, x, t):
        x = self.conv_in(x)
        t_emb = get_time_embedding(t, self.t_emb_dim)

        down_outs = []
        
        for i, down_block in enumerate(self.down_blocks):
            down_outs.append(x)
            x = down_block(x, t_emb=t_emb)

        for i, mid_block in enumerate(self.mid_blocks):
            x = mid_block(x, t_emb=t_emb)

        for i, up_block in enumerate(self.up_blocks):
            down_out = down_outs.pop()
            x = up_block(x, t_emb=t_emb, down_out=down_out)

        x = self.out_block(x)
        return x
    
    
if __name__ == "__main__":
    import torch_directml
    device = torch_directml.device()
    
    # Example usage
    unet = UNet(
        in_channels=4,
        out_channels=4,
        down_channels=[256, 384, 512, 768],
        mid_channels=[768, 512],
        t_emb_dim=512,
        downsamples=[True, True, True],
        num_down_layers=2,
        num_mid_layers=2,
        num_up_layers=2,
        self_attn=[True, True, True],
        num_heads=4,
        conv_out_channels=128
    ).to(device)
    x = torch.randn(2, 4, 32, 32, device=device)  # Example input tensor
    t = torch.randint(0, 1000, (2,), device=device)  # Example time steps tensor
    x = x.to(device)
    out = unet(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
        