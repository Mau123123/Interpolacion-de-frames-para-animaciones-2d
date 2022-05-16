from tkinter import X
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.conv(x)

class resNet(nn.Module):
    """The quadratic model"""
    def __init__(self, in_channels, out_channels):
        super(resNet, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv12 = ConvBlock(8, 8)
        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv22 = ConvBlock(16, 16)
        self.middle1 =nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.middle2 = ConvBlock(32, 32)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = ConvBlock(16, 16)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(16, 16)
        self.finalConv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.Hardsigmoid()

    def forward(self, x):
        xold = x.clone()
        
        x = self.conv11(x)
        x1 = torch.utils.checkpoint.checkpoint(self.conv12,x)
        x = torch.cat([x,x1], dim = 1)
        x = self.down(x)
        
        x = torch.utils.checkpoint.checkpoint(self.conv21,x)
        x1 = torch.utils.checkpoint.checkpoint(self.conv22,x)
        x = torch.cat([x,x1], dim = 1)
        x = self.down(x)
        
        
        x = torch.utils.checkpoint.checkpoint(self.middle1,x)
        x1 = torch.utils.checkpoint.checkpoint(self.middle2,x)
        x = torch.cat([x,x1], dim = 1)
        
       
        x = torch.utils.checkpoint.checkpoint(self.up2,x)
        
        x = torch.utils.checkpoint.checkpoint(self.up3,x)
        
        if x.shape != xold.shape:
                x = TF.resize(x, size=xold.shape[2:])
        x = self.finalConv(x)
        return self.activation(x)
