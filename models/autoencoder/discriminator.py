import torch
import torch.nn as nn

from .block import Block
from .utils import format_input

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            Block(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(256, 1),
        )
        
        # 8x8 image size
        
    def forward(self, x):
        x = format_input(x)
        x = self.discriminator(x)
        return x
    
        

    