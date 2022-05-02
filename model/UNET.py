import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
import torchvision.transforms.functional as TF
try:
    from .convBlock import ConvBlock                  
except ImportError:
    try:
        from convBlock import ConvBlock
    except ImportError:
        print("no existe modulo")

class UNET(nn.Module):
    
    def __init__(
            self, in_channels=3, out_channels=3, channels=[15, 30, 60, 120],
    ):
        super(UNET, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(in_channels)
        self.featureExtraction = nn.ModuleList()
        self.featureExtraction.requires_grad = False
        
        for feature in channels:
            self.encoders.append(ConvBlock(in_channels, feature))
            in_channels = feature
            
        in_channels = 3
        for feature in channels:
            out = int(feature)
            self.featureExtraction.append(ConvBlock(in_channels, out))
            in_channels = out

        for feature in reversed(channels):
            self.decoders.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.decoders.append(ConvBlock(feature*4, feature))

        self.middle = ConvBlock(channels[-1], channels[-1]*2)
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)


    def forward(self, x1, x2, x):
        skip_connections = []
        features = []
        x11 = self.norm(x1)
        x22 = self.norm(x2)
        x = self.norm(x)
        with torch.no_grad():
            for feature in self.featureExtraction:
                x11 =  feature(x11)
                x22 =  feature(x22)
                features.append( torch.cat((x11, x22), dim=1))
                x11 = self.pool(x11)
                x22 = self.pool(x22)
        for encoder in self.encoders:
            x = checkpoint(encoder,x)
            skip_connections.append(x)
            x = self.pool(x)

        x = x = checkpoint(self.middle,x)
        skip_connections = skip_connections[::-1]
        features = features[::-1]

        for idx in range(0, len(self.decoders), 2):
            x = checkpoint(self.decoders[idx],x)
            skip_connection = skip_connections[idx//2]
            feature = features[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x, feature), dim=1)
            x = checkpoint(self.decoders[idx+1],concat_skip)

        return self.final_conv(x)