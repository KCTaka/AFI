import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        init_x = x
        x = self.block1(x)
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
    def __init__(self, in_channels, out_channels,
                 downsample, num_heads,
                 num_layers, use_self_attention):
        super(DownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_self_attention = use_self_attention
        
        self.model = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.model.append(ResNetBlock(in_channels, out_channels))
            else:
                self.model.append(ResNetBlock(out_channels, out_channels))
                
            if use_self_attention:
                self.model.append(SelfAttentionBlock(out_channels, num_heads))
                
        if downsample:
            self.model.append(nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))
            
    def forward(self, x):
        for layer in self.model:
            if isinstance(layer, SelfAttentionBlock):
                x = layer(x) + x
            else:
                x = layer(x)
        return x
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super(MidBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.model = nn.ModuleList()
        self.model.append(ResNetBlock(in_channels, out_channels))

        for i in range(num_layers):
            
            self.model.append(SelfAttentionBlock(out_channels, num_heads))
            
            self.model.append(ResNetBlock(out_channels, out_channels))

    def forward(self, x):
        for layer in self.model:
            if isinstance(layer, ResNetBlock):
                x = layer(x)
            elif isinstance(layer, SelfAttentionBlock):
                x = layer(x) + x
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 upsample, num_heads,
                 num_layers, use_self_attention):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_self_attention = use_self_attention
        
        self.model = nn.ModuleList()
        
        if upsample:
            self.model.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1))
        
        for i in range(num_layers):
            if i == 0:
                self.model.append(ResNetBlock(in_channels, out_channels))
            else:
                self.model.append(ResNetBlock(out_channels, out_channels))
                
            if use_self_attention:
                self.model.append(SelfAttentionBlock(out_channels, num_heads))
                
    def forward(self, x):
        for layer in self.model:
            if isinstance(layer, SelfAttentionBlock):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

if __name__ == "__main__":
    # Example usage
    x = torch.randn(4, 3, 64, 64)  # Example input tensor
    down_block = DownBlock(3, 64, downsample=True, num_heads=4, num_layers=2, use_self_attention=True)
    output = down_block(x)
    print(output.shape)  
    mid_block = MidBlock(64, 128, num_heads=4, num_layers=2)
    output = mid_block(output)
    print(output.shape)
    up_block = UpBlock(128, 64, upsample=True, num_heads=4, num_layers=2, use_self_attention=True)
    output = up_block(output)
    print(output.shape)
