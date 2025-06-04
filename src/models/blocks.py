import autoroot
import autorootcwd

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def get_time_embedding(t: torch.Tensor, embedding_dim: int, max_period: int = 10000):
    half = embedding_dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / (half - 1) )
    # [batch, half]
    args = t[:, None] * freqs[None, :]  # [batch, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
        
    return emb  # [batch, embedding_dim]

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=None):
        super(ResNetBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        
        self.time_embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels),
        ) if t_emb_dim is not None else None

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x, t_emb=None):
        init_x = x
        x = self.block1(x)

        if self.time_embedding_layer is not None and t_emb is not None:
            t_emb = self.time_embedding_layer(t_emb)
            x = x + t_emb.view(x.size(0), x.size(1), 1, 1)
        
        x = self.block2(x)
        
        # Apply skip connection with channel dimension matching
        init_x = self.skip_conv(init_x)

        return x + init_x
    
class SelfAttentionBlock(nn.Module): 
    def __init__(self, in_channels, num_heads):
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
    
class DownBlock(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 t_emb_dim=None,
                 downsample=False, num_heads=4,
                 num_layers=1, use_self_attention=False):
        super(DownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.downsample = downsample
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_self_attention = use_self_attention
        
        self.model = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.model.append(ResNetBlock(in_channels, out_channels, t_emb_dim=t_emb_dim))
            else:
                self.model.append(ResNetBlock(out_channels, out_channels, t_emb_dim=t_emb_dim))

            if use_self_attention:
                self.model.append(SelfAttentionBlock(out_channels, num_heads))
                
        if downsample:
            self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x, t_emb):
        for layer in self.model:
            if isinstance(layer, SelfAttentionBlock):
                x = layer(x) + x
            if isinstance(layer, ResNetBlock):
                x = layer(x, t_emb=t_emb)
            else:
                x = layer(x)
        
        if self.downsample:
            x = self.down_conv(x)

        return x

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=None, num_heads=4, num_layers=1):
        super(MidBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.model = nn.ModuleList()
        self.model.append(ResNetBlock(in_channels, out_channels, t_emb_dim=t_emb_dim))

        for i in range(num_layers):
            
            self.model.append(SelfAttentionBlock(out_channels, num_heads))

            self.model.append(ResNetBlock(out_channels, out_channels, t_emb_dim=t_emb_dim))

    def forward(self, x, t_emb=None):
        for layer in self.model:
            if isinstance(layer, ResNetBlock):
                x = layer(x, t_emb=t_emb)
            elif isinstance(layer, SelfAttentionBlock):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=None,
                 upsample=False, num_heads=4,
                 num_layers=1, use_self_attention=False,
                 require_down_cat=False):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.upsample = upsample
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_self_attention = use_self_attention
        self.require_down_cat = require_down_cat

        if upsample:
            eff_channels = in_channels // 2 if require_down_cat else in_channels
            self.up_conv = nn.ConvTranspose2d(eff_channels, eff_channels, kernel_size=4, stride=2, padding=1)

        self.model = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.model.append(ResNetBlock(in_channels, out_channels, t_emb_dim=t_emb_dim))
            else:
                self.model.append(ResNetBlock(out_channels, out_channels, t_emb_dim=t_emb_dim))

            if use_self_attention:
                self.model.append(SelfAttentionBlock(out_channels, num_heads))

    def forward(self, x, t_emb=None, down_out=None):
        
        if self.upsample:
            x = self.up_conv(x)

        if down_out is not None:
            x = torch.cat([x, down_out], dim=1)

        for layer in self.model:
            if isinstance(layer, SelfAttentionBlock):
                x = layer(x) + x
            elif isinstance(layer, ResNetBlock):
                x = layer(x, t_emb=t_emb)
            else:
                x = layer(x)
        return x

if __name__ == "__main__":
    # Example usage
    import torch_directml
    device = torch_directml.device()
    x = torch.randn(4, 3, 64, 64).to(device)  # Example input tensor
    t_emb = torch.randn(4, 32).to(device)  # Example time embedding tensor
    down_block = DownBlock(3, 64, downsample=True, num_heads=4, num_layers=2, use_self_attention=True).to(device)
    output = down_block(x, t_emb=t_emb)
    print(output.shape)
    mid_block = MidBlock(64, 128, t_emb_dim=32, num_heads=4, num_layers=2).to(device)
    output = mid_block(output, t_emb=t_emb)
    print(output.shape)
    up_block = UpBlock(128, 64, upsample=True, num_heads=4, num_layers=2, use_self_attention=False).to(device)
    output = up_block(output, t_emb=t_emb)
    print(output.shape)
