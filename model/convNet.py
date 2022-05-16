import torch.nn as nn
import torch.utils.checkpoint as checkpoint
try:
    from .convBlock import ConvBlock                  
except ImportError:
    try:
        from convBlock import ConvBlock
    except ImportError:
        print("no existe modulo")

class convNet(nn.Module):
    """The quadratic model"""
    def __init__(self, in_channels, out_channels):
        super(convNet, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv12 = ConvBlock(16, 16)
        self.conv2 = ConvBlock(16, 32)
        self.middle1 =nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = ConvBlock(32, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(16, 16)
        self.finalConv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.Hardsigmoid()

    def forward(self, x):
        x = self.conv11(x)
        x = checkpoint.checkpoint(self.conv12,x)
        x = self.down(x)
        x = checkpoint.checkpoint(self.conv2,x)
        x = self.down(x)
        x = self.middle1(x)
        x = self.up2(x)
        x = checkpoint.checkpoint(self.conv5,x)
        x = self.up3(x)
        x = self.conv6(x)
        x = self.finalConv(x)
        x = self.activation(x)
        return x