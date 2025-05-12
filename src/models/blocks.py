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