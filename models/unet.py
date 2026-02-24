"""
Patch-Diffusion CNN (PD) — Phase 1 model from KernelFusion (arXiv:2503.21907).

This is a pure convolutional network with NO pooling, NO strides, and NO attention.
It has a restricted receptive field of 15×15 pixels to learn local patch statistics
from a single image.

Architecture (from paper Section 4.4 + Appendix A1.1):
  - 1 block of two 3×3 convolutions
  - 6 blocks of (3×3 + 1×1) convolutions
  - 128 hidden channels
  - Timestep conditioning via sinusoidal embedding injected into each block
  - Single-channel input/output (adapted for 16-bit grayscale SEM images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.diffusion import get_sinusoidal_embedding


class TimestepConditionedBlock(nn.Module):
    """A convolutional block conditioned on the diffusion timestep."""
    
    def __init__(self, in_channels, out_channels, t_emb_dim=128, use_1x1=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=1 if use_1x1 else 3,
                               padding=0 if use_1x1 else 1)
        
        # Timestep projection: project embedding to channel dimension for conditioning
        self.t_proj = nn.Linear(t_emb_dim, out_channels)
        
        self.act = nn.GELU()
    
    def forward(self, x, t_emb):
        """
        Args:
            x: [B, C, H, W]
            t_emb: [B, t_emb_dim] sinusoidal timestep embedding
        """
        h = self.act(self.conv1(x))
        
        # Add timestep conditioning (broadcast over spatial dims)
        t_scale = self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        h = h + t_scale
        
        h = self.act(self.conv2(h))
        return h


class PatchDiffusionCNN(nn.Module):
    """
    Patch-Diffusion model for learning internal patch statistics of a single image.
    
    Pure CNN with 15×15 receptive field:
      - Block 0: two 3×3 convolutions (contributes 4 to RF: 5×5 after this block)
      - Blocks 1-6: 3×3 + 1×1 convolutions each (each contributes +2 to RF)
      - Total RF: 5 + 6*2 - 2 = 15×15 ✓  (accounting for shared boundaries)
    
    All blocks are conditioned on timestep t via sinusoidal embedding.
    """
    
    def __init__(self, in_channels=1, hidden_channels=128, t_emb_dim=128):
        super().__init__()
        
        self.t_emb_dim = t_emb_dim
        
        # Block 0: Input block with two 3×3 convolutions
        self.input_block = TimestepConditionedBlock(
            in_channels, hidden_channels, t_emb_dim, use_1x1=False
        )
        
        # Blocks 1-6: Each has 3×3 + 1×1 convolutions
        self.blocks = nn.ModuleList([
            TimestepConditionedBlock(hidden_channels, hidden_channels, t_emb_dim, use_1x1=True)
            for _ in range(6)
        ])
        
        # Output projection: 1×1 conv to map back to image channels
        self.output_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
    
    def forward(self, x, t):
        """
        Predict velocity v_t given noised image x_t and timestep t.
        
        Args:
            x: Noised image patches [B, 1, H, W]
            t: Timestep indices [B] (integer tensor)
        
        Returns:
            v_pred: Predicted velocity [B, 1, H, W]
        """
        # Compute sinusoidal timestep embedding
        t_emb = get_sinusoidal_embedding(t, self.t_emb_dim).to(x.device)
        
        # Forward through blocks
        h = self.input_block(x, t_emb)
        
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output projection
        v_pred = self.output_conv(h)
        return v_pred