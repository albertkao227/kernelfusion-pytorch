"""
Deep Image Prior (DIP) U-Net for Phase 2 of KernelFusion.

Provides global structural coherence during the reverse diffusion process.
The PD model only has a 15×15 local receptive field, so this DIP U-Net
is essential for maintaining global image structure.

Architecture (from paper Appendix A1.2):
  - 5-level encoder-decoder
  - Channel progression: 32 → 64 → 128 → 256 → 512
  - Each block: 2× Conv2d(3×3) + BatchNorm + ReLU
  - Skip connections at each level
  - Tanh output activation (ensures output in [-1, 1])
  - Single-channel input/output (16-bit grayscale SEM images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3×3 convolutions with BatchNorm and ReLU."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class DIPUNet(nn.Module):
    """
    Deep Image Prior U-Net for global structure preservation.
    
    Trained from scratch during Phase 2 reverse diffusion.
    At each timestep t, it refines the HR prediction to maintain
    global coherence that the local PD model cannot provide.
    """
    
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        
        ch = base_channels  # 32
        
        # Encoder (downsampling path)
        self.enc1 = ConvBlock(in_channels, ch)       # 32
        self.enc2 = ConvBlock(ch, ch * 2)             # 64
        self.enc3 = ConvBlock(ch * 2, ch * 4)         # 128
        self.enc4 = ConvBlock(ch * 4, ch * 8)         # 256
        
        # Bottleneck
        self.bottleneck = ConvBlock(ch * 8, ch * 16)  # 512
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 8)        # 256 (concat with skip)
        
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4)         # 128
        
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2)         # 64
        
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)             # 32
        
        # Output: 1×1 conv + Tanh to ensure [-1, 1] range
        self.output = nn.Sequential(
            nn.Conv2d(ch, in_channels, kernel_size=1),
            nn.Tanh(),
        )
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        """
        Args:
            x: Image tensor [B, 1, H, W]
        
        Returns:
            out: Refined image [B, 1, H, W] in [-1, 1]
        """
        # Store original size for padding if needed
        _, _, H, W = x.shape
        
        # Pad to multiple of 16 for 4 levels of downsampling
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        out = self.output(d1)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        return out
