import torch
import torch.nn as nn

from src.utils.helpers import format_input

class Block(nn.Module):
    '''    # Block architecture
    # - Convolutional layers with ReLU activation and Batch Normalization
    # - Input -> Convolution -> ReLU -> BatchNorm -> Convolution -> ReLU -> BatchNorm -> Output
    # - Each block reduces HxW by half and changes the number of channels'''
    
    def __init__(self, in_channels, out_channels, skip=True):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Skip connection for channel dimension matching
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels and skip else None
        self.activation = nn.LeakyReLU(0.2)
        
        self.skip = skip
    
    def forward(self, x):
        init_x = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # Apply skip connection with dimension matching
        if self.skip_conv is not None:
            init_x = self.skip_conv(init_x)
            
        x = self.activation(x + init_x if self.skip else x)
        
        return x

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
    
        

    