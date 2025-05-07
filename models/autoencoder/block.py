import torch
import torch.nn as nn
import torch.nn.functional as F


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