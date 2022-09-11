import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
     
    def forward(self, x):
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.conv_block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.conv_block(x1)

        return x2

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
            super().__init__()

            self.upscale = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv_block = ConvBlock(in_channels, out_channels)

    
    def forward(self, x, enc_x):
        x1 = self.upscale(x)

        delta_Y = enc_x.size()[2] - x1.size()[2]
        delta_X = enc_x.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [delta_X // 2, delta_X - delta_X // 2,
                        delta_Y // 2, delta_Y - delta_Y // 2])
        
        x1 = torch.cat([enc_x,x1], dim=1)

        x2 = self.conv_block(x1)

        return x2 
    
class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        
        self.enc0= ConvBlock(in_channels, 64)
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)
        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc0(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        logits = self.outc(x)
        return logits
