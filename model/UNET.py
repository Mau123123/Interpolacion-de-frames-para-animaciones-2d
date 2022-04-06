import torch
import torch.nn as nn
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
            self, in_channels=6, out_channels=3, channels=[32, 64],
    ):
        super(UNET, self).__init__()
        self.conv = nn.Conv2d(3, 8, 3, 1, 1, bias=False)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        in_channels = 8
        for feature in channels:
            self.encoders.append(ConvBlock(in_channels, feature))
            in_channels = feature

        for feature in reversed(channels):
            self.decoders.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.decoders.append(ConvBlock(feature*2, feature))

        self.middle = ConvBlock(channels[-1], channels[-1]*2)
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)


    def forward(self, x1, x2):
        skip_connections = []
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x = torch.cat((x1,x2),1)
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.middle(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[idx+1](concat_skip)

        return self.final_conv(x)