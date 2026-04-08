import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class DIPUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        ch = base_channels
        
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.enc4 = ConvBlock(ch * 4, ch * 8)
        
        self.bottleneck = ConvBlock(ch * 8, ch * 16)
        
        self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 8)
        
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4)
        
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2)
        
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)
        
        self.output = nn.Sequential(
            nn.Conv2d(ch, out_channels, kernel_size=1),
            nn.Tanh(), # Ensures output is in [-1, 1] for diffusion consistency
        )
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        _, _, H, W = x.shape
        # Pad to multiple of 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = self.output(d1)
        
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out