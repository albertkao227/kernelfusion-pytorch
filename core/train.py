"""
Phase 1: Train the Patch-Diffusion (PD) model on a single LR image.

Learns the internal patch distribution by training a velocity-prediction
DDPM on random 64×64 crops from the input image.

Reference: KernelFusion (arXiv:2503.21907), Section 4.1 + Appendix A1.1
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import SEMPatchDataset
from models.unet import PatchDiffusionCNN
from core.diffusion import DDPMSchedule, forward_diffusion, compute_velocity_target


def train_internal_diffusion(
    image_path,
    total_steps=600_000,
    batch_size=32,
    patch_size=64,
    lr=1e-4,
    T=1000,
    hidden_channels=128,
    device='cuda',
    log_every=5000,
):
    """
    Phase 1: Train the internal Patch-Diffusion model on a single SEM image.
    
    This model learns the unique statistical distributions of the image's patches
    to act as an anti-hallucination prior in Phase 2.
    
    Args:
        image_path: Path to the 16-bit grayscale SEM image
        total_steps: Total training gradient steps (paper: 600,000)
        batch_size: Batch size of random crops per step
        patch_size: Size of random crops (paper: 64×64)
        lr: Learning rate (paper: 1e-4)
        T: Number of DDPM timesteps (paper: 1000)
        hidden_channels: Hidden channels in PD model (paper: 128)
        device: Device to train on
        log_every: Log loss every N steps
    
    Returns:
        model: Trained PatchDiffusionCNN
        schedule: DDPMSchedule used during training (needed for Phase 2)
    """
    # 1. Initialize the dataset (loads image, provides random crops)
    dataset = SEMPatchDataset(image_path, patch_size=patch_size, mode='random')
    
    # We use a DataLoader that re-samples random crops each iteration
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    
    # 2. Initialize model and DDPM schedule
    model = PatchDiffusionCNN(
        in_channels=1,
        hidden_channels=hidden_channels,
    ).to(device)
    
    schedule = DDPMSchedule(T=T).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    
    model.train()
    print(f"Phase 1: Training PD model for {total_steps:,} steps on device={device}")
    print(f"  Patch size: {patch_size}×{patch_size}, Batch size: {batch_size}")
    print(f"  Hidden channels: {hidden_channels}, DDPM timesteps T={T}")
    
    # 3. Training loop — iterate until total_steps reached
    step = 0
    running_loss = 0.0
    
    while step < total_steps:
        for patches in dataloader:
            if step >= total_steps:
                break
            
            patches = patches.to(device)  # [B, 1, patch_size, patch_size]
            
            # Sample random discrete timesteps t ∈ {0, 1, ..., T-1}
            t = torch.randint(0, schedule.T, (patches.shape[0],), device=device)
            
            # Forward diffusion: q(x_t | x_0)
            x_t, epsilon = forward_diffusion(patches, t, schedule)
            
            # Compute velocity target: v_t = √ᾱ_t · ε − √(1−ᾱ_t) · x_0
            v_target = compute_velocity_target(patches, epsilon, t, schedule)
            
            # Predict velocity with PD model
            v_pred = model(x_t, t)
            
            # Velocity prediction loss
            loss = F.mse_loss(v_pred, v_target)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            step += 1
            
            # Logging
            if step % log_every == 0:
                avg_loss = running_loss / log_every
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Step [{step:>7,}/{total_steps:,}] | "
                      f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
                running_loss = 0.0
    
    print("Phase 1 training complete.")
    return model, schedule