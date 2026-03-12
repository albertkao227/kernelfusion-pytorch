"""
Phase 1: Train the Patch-Diffusion (PD) model on a single LR image.

Learns the internal patch distribution by training a velocity-prediction
DDPM on random 64×64 crops from the input image.

Reference: KernelFusion (arXiv:2503.21907), Section 4.1 + Appendix A1.1

Usage:
    python -m core.train --input data/images/lena_gray16.tiff
    python -m core.train --input data/images/lena_gray16.tiff --total-steps 500 --batch-size 8
"""

import argparse
import itertools
import os
import sys

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Adjust path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import SEMPatchDataset
from models.unet import PatchDiffusionCNN
from core.diffusion import (
    DDPMSchedule,
    forward_diffusion,
    compute_velocity_target,
    velocity_to_x0,
    ddpm_reverse_step,
)


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------
class EMAModel:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model):
        """Copy EMA weights into model (for inference / sampling)."""
        backup = {}
        for name, p in model.named_parameters():
            backup[name] = p.data.clone()
            p.data.copy_(self.shadow[name])
        return backup

    def restore(self, model, backup):
        """Restore original weights after EMA inference."""
        for name, p in model.named_parameters():
            p.data.copy_(backup[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Sample generation (for monitoring)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(model, schedule, n_samples=4, patch_size=64, steps=50, device="cpu"):
    """
    Generate sample patches via a short reverse-diffusion chain.

    Returns a tensor of shape [n_samples, 1, patch_size, patch_size] in [-1, 1].
    """
    model.eval()

    # Start from pure noise
    x = torch.randn(n_samples, 1, patch_size, patch_size, device=device)

    # Subsample timesteps uniformly for faster sampling
    ts = torch.linspace(schedule.T - 1, 0, steps, device=device).long()

    for t_val in ts:
        t_batch = t_val.expand(n_samples)
        v_pred = model(x, t_batch)
        x_0_pred = velocity_to_x0(v_pred, x, t_batch, schedule)
        x_0_pred.clamp_(-1, 1)
        x = ddpm_reverse_step(x, x_0_pred, int(t_val.item()), schedule)

    model.train()
    return x.clamp(-1, 1)


def save_sample_grid(samples, path, cols=4):
    """Save a grid of sample patches as a single image."""
    import numpy as np
    import cv2

    n = samples.shape[0]
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    patches = []
    for i in range(n):
        # [-1,1] → [0, 65535] uint16
        img = samples[i, 0].cpu().numpy()
        img = ((img + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)
        patches.append(img)

    # Pad to fill grid
    h, w = patches[0].shape
    while len(patches) < rows * cols:
        patches.append(np.zeros((h, w), dtype=np.uint16))

    rows_imgs = []
    for r in range(rows):
        rows_imgs.append(np.concatenate(patches[r * cols:(r + 1) * cols], axis=1))
    grid = np.concatenate(rows_imgs, axis=0)

    cv2.imwrite(path, grid)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, ema, optimizer, scheduler, schedule, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "schedule_T": schedule.T,
    }, path)


def load_checkpoint(path, model, ema, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_internal_diffusion(
    image_path,
    total_steps=600_000,
    batch_size=32,
    patch_size=64,
    lr=1e-4,
    T=1000,
    hidden_channels=128,
    device="cuda",
    log_every=500,
    save_every=50_000,
    sample_every=10_000,
    output_dir=".",
    resume=None,
):
    """
    Phase 1: Train the internal Patch-Diffusion model on a single image.

    Args:
        image_path: Path to the 16-bit grayscale image
        total_steps: Total training gradient steps (paper: 600,000)
        batch_size: Batch size of random crops per step
        patch_size: Size of random crops (paper: 64×64)
        lr: Learning rate (paper: 1e-4)
        T: Number of DDPM timesteps (paper: 1000)
        hidden_channels: Hidden channels in PD model (paper: 128)
        device: Device to train on
        log_every: Update progress bar every N steps
        save_every: Save checkpoint every N steps
        sample_every: Generate sample patches every N steps
        output_dir: Base directory for checkpoints/ and samples/
        resume: Path to checkpoint to resume from

    Returns:
        model: Trained PatchDiffusionCNN
        schedule: DDPMSchedule used during training
    """
    # Directories
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    dataset = SEMPatchDataset(image_path, patch_size=patch_size, mode="random")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )

    # 2. Model + Schedule
    model = PatchDiffusionCNN(
        in_channels=1, hidden_channels=hidden_channels,
    ).to(device)
    schedule = DDPMSchedule(T=T).to(device)

    # 3. Optimizer + LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    # 4. EMA
    ema = EMAModel(model, decay=0.9999)

    # 5. Resume from checkpoint
    start_step = 0
    if resume and os.path.isfile(resume):
        start_step = load_checkpoint(resume, model, ema, optimizer, scheduler, device)
        print(f"Resumed from checkpoint at step {start_step:,}")

    # 6. Training loop
    model.train()
    infinite_loader = itertools.cycle(dataloader)

    pbar = tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        desc="Phase 1",
        unit="step",
        dynamic_ncols=True,
    )

    running_loss = 0.0
    loss_count = 0

    for step in pbar:
        patches = next(infinite_loader).to(device)  # [B, 1, P, P]

        # Sample random timesteps
        t = torch.randint(0, schedule.T, (patches.shape[0],), device=device)

        # Forward diffusion
        x_t, epsilon = forward_diffusion(patches, t, schedule)

        # Velocity target
        v_target = compute_velocity_target(patches, epsilon, t, schedule)

        # Model prediction
        v_pred = model(x_t, t)

        # Loss
        loss = F.mse_loss(v_pred, v_target)

        # Backprop with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # EMA update
        ema.update(model)

        # Tracking
        running_loss += loss.item()
        loss_count += 1

        if (step + 1) % log_every == 0:
            avg = running_loss / loss_count
            pbar.set_postfix(loss=f"{avg:.5f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0.0
            loss_count = 0

        # Checkpoint
        if (step + 1) % save_every == 0:
            path = os.path.join(ckpt_dir, f"pd_step{step+1}.pt")
            save_checkpoint(path, model, ema, optimizer, scheduler, schedule, step + 1)
            pbar.write(f"  ✓ Checkpoint saved → {path}")

        # Sample generation (using EMA weights)
        if (step + 1) % sample_every == 0:
            backup = ema.apply(model)
            samples = generate_samples(
                model, schedule, n_samples=4, patch_size=patch_size, device=device,
            )
            ema.restore(model, backup)
            path = os.path.join(sample_dir, f"samples_step{step+1}.tif")
            save_sample_grid(samples, path)
            pbar.write(f"  ✓ Samples saved → {path}")

    # Final save
    final_path = os.path.join(ckpt_dir, "pd_final.pt")
    save_checkpoint(final_path, model, ema, optimizer, scheduler, schedule, total_steps)
    print(f"Phase 1 training complete. Final checkpoint → {final_path}")

    # Return EMA model for downstream use
    ema.apply(model)
    return model, schedule


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 1: Train Patch-Diffusion model")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to 16-bit grayscale image")
    parser.add_argument("--total-steps", type=int, default=600_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--hidden-ch", type=int, default=128)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/mps/cpu). Auto-detected if omitted.")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Image:  {args.input}")

    train_internal_diffusion(
        image_path=args.input,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        lr=args.lr,
        T=args.T,
        hidden_channels=args.hidden_ch,
        device=device,
        log_every=args.log_every,
        save_every=args.save_every,
        sample_every=args.sample_every,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()