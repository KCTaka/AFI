import autoroot
import autorootcwd

import torch
import torch.nn as nn

from src.utils.formats import format_input

class Discriminator(nn.Module):
    def __init__(self, in_channels=3,
                 conv_channels=[64, 128, 256]):
        super(Discriminator, self).__init__()
        self.conv_channels = conv_channels
        self.in_channels = in_channels
        self.out_channels = 1
        
        self.activation = nn.LeakyReLU(0.2)
        self.model = nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels, conv_channels[0], bias=False, kernel_size=4, stride=2, padding=1))
        self.model.append(self.activation)
        
        for i in range(1, len(conv_channels)):
            self.model.append(nn.Conv2d(conv_channels[i-1], conv_channels[i], kernel_size=4, stride=2, padding=1))
            self.model.append(nn.BatchNorm2d(conv_channels[i]))
            self.model.append(self.activation)

        self.model.append(nn.Conv2d(conv_channels[-1], self.out_channels, kernel_size=4, stride=1, padding=1))

    def forward(self, x):
        x = format_input(x)
        for layer in self.model:
            x = layer(x)
        return x
    
if __name__ == "__main__":
    # Example usage
    x = torch.randn(4, 3, 256, 256)  # Example input tensor
    model = Discriminator()
    output = model(x)
    print(output.shape)  # Should be (4, 1, 8, 8) for the given input size
    print(model)
    
        

    