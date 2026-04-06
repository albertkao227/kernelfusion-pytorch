#!/usr/bin/env python3
"""
eval_pd.py — Evaluate a trained Patch-Diffusion (PD) model
============================================================

Creates side-by-side visual comparisons to assess PD model quality:

  1. DENOISING TEST:  real patch → add noise → PD denoises → compare
  2. GENERATION TEST: pure noise → PD generates → compare with real patches

Usage
-----
    # Using a saved checkpoint
    python eval_pd.py --input data/images/lena_gray16.tiff \
                      --checkpoint checkpoints/pd_step50000.pt

    # Quick training + eval (train a small model on the spot)
    python eval_pd.py --input data/images/lena_gray16.tiff \
                      --quick-train 5000

    # Customize noise levels for denoising test
    python eval_pd.py --input data/images/lena_gray16.tiff \
                      --checkpoint checkpoints/pd_final.pt \
                      --noise-levels 50 200 500 800
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import SEMPatchDataset
from models.unet import PatchDiffusionCNN
from core.diffusion import (
    DDPMSchedule,
    forward_diffusion,
    compute_velocity_target,
    velocity_to_x0,
    ddpm_reverse_step,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_to_uint16(t: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] tensor to a 16-bit uint16 numpy array."""
    img = t.detach().cpu().squeeze().numpy()
    img = ((img + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)
    return img


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] tensor to 8-bit uint8 for display."""
    img = t.detach().cpu().squeeze().numpy()
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return img


def add_label(img: np.ndarray, text: str, font_scale: float = 0.5) -> np.ndarray:
    """Add a text label to the top of a uint8 image."""
    h, w = img.shape[:2]
    # Create label bar
    bar_h = 22
    bar = np.zeros((bar_h, w), dtype=np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 255, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def extract_fixed_patches(
    image_path: str, patch_size: int, n_patches: int, device: str = "cpu",
) -> torch.Tensor:
    """Extract evenly-spaced patches from the image for reproducible evaluation."""
    from utils.image_utils import load_sem_16bit
    img = load_sem_16bit(image_path, device=device)  # [1, 1, H, W]
    _, _, H, W = img.shape

    patches = []
    # Grid positions (avoid edges)
    margin = patch_size
    ys = np.linspace(margin, H - margin - patch_size, max(1, int(np.sqrt(n_patches))))
    xs = np.linspace(margin, W - margin - patch_size, max(1, int(np.sqrt(n_patches))))

    for y in ys:
        for x in xs:
            y_i, x_i = int(y), int(x)
            patch = img[:, :, y_i:y_i + patch_size, x_i:x_i + patch_size]
            patches.append(patch)
            if len(patches) >= n_patches:
                break
        if len(patches) >= n_patches:
            break

    # If not enough from grid, fill with random
    while len(patches) < n_patches:
        y_i = np.random.randint(0, H - patch_size)
        x_i = np.random.randint(0, W - patch_size)
        patches.append(img[:, :, y_i:y_i + patch_size, x_i:x_i + patch_size])

    return torch.cat(patches, dim=0).to(device)  # [N, 1, P, P]


# ═══════════════════════════════════════════════════════════════════════════
# Core evaluation functions
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_patch(
    model: PatchDiffusionCNN,
    schedule: DDPMSchedule,
    clean_patch: torch.Tensor,
    noise_level: int,
    sampling_steps: int = 50,
) -> torch.Tensor:
    """
    Add noise to a clean patch at a given timestep, then denoise it.

    Returns the denoised patch.
    """
    model.eval()
    device = clean_patch.device

    # Forward diffusion: add noise to the clean patch
    t = torch.tensor([noise_level], device=device)
    x_t, _ = forward_diffusion(clean_patch.unsqueeze(0), t, schedule)

    # Reverse diffusion: denoise step by step from noise_level to 0
    ts = torch.linspace(noise_level, 0, min(sampling_steps, noise_level + 1),
                        device=device).long()
    ts = ts.unique()  # remove duplicates

    x = x_t
    for t_val in ts:
        t_batch = t_val.unsqueeze(0)
        v_pred = model(x, t_batch)
        x0_pred = velocity_to_x0(v_pred, x, t_batch, schedule).clamp(-1, 1)
        if t_val > 0:
            x = ddpm_reverse_step(x, x0_pred, int(t_val.item()), schedule)
        else:
            x = x0_pred

    return x.squeeze(0).clamp(-1, 1)


@torch.no_grad()
def generate_from_noise(
    model: PatchDiffusionCNN,
    schedule: DDPMSchedule,
    n_samples: int = 4,
    patch_size: int = 64,
    sampling_steps: int = 100,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate fresh patches from pure Gaussian noise."""
    model.eval()
    x = torch.randn(n_samples, 1, patch_size, patch_size, device=device)
    ts = torch.linspace(schedule.T - 1, 0, sampling_steps, device=device).long()

    for t_val in ts:
        t_batch = t_val.expand(n_samples)
        v_pred = model(x, t_batch)
        x0_pred = velocity_to_x0(v_pred, x, t_batch, schedule).clamp(-1, 1)
        x = ddpm_reverse_step(x, x0_pred, int(t_val.item()), schedule)

    return x.clamp(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Build comparison images
# ═══════════════════════════════════════════════════════════════════════════

def build_denoise_comparison(
    model, schedule, patches, noise_levels, device,
) -> np.ndarray:
    """
    Build a grid: rows = different patches, columns = Original | Noised@t | Denoised@t | ...

    Returns a uint8 image.
    """
    rows = []

    for p_idx in range(patches.shape[0]):
        patch = patches[p_idx]  # [1, P, P]
        row_imgs = [add_label(tensor_to_uint8(patch), "Original")]

        for t_noise in noise_levels:
            # Forward diffusion (noised)
            t_tensor = torch.tensor([t_noise], device=device)
            x_noised, _ = forward_diffusion(patch.unsqueeze(0), t_tensor, schedule)
            noised_img = tensor_to_uint8(x_noised.squeeze(0))

            # Denoise
            denoised = denoise_patch(model, schedule, patch, t_noise)
            denoised_img = tensor_to_uint8(denoised)

            row_imgs.append(add_label(noised_img, f"Noised t={t_noise}"))
            row_imgs.append(add_label(denoised_img, f"Denoised t={t_noise}"))

        rows.append(np.hstack(row_imgs))

    return np.vstack(rows)


def build_generation_comparison(
    model, schedule, real_patches, n_generated, patch_size, device,
) -> np.ndarray:
    """
    Build a grid: top row = real patches, bottom row = generated patches.

    Returns a uint8 image.
    """
    n_show = min(n_generated, real_patches.shape[0])

    # Real patches row
    real_imgs = []
    for i in range(n_show):
        real_imgs.append(add_label(tensor_to_uint8(real_patches[i]), f"Real #{i+1}"))
    real_row = np.hstack(real_imgs)

    # Generated patches row
    generated = generate_from_noise(
        model, schedule, n_samples=n_show, patch_size=patch_size,
        sampling_steps=100, device=device,
    )
    gen_imgs = []
    for i in range(n_show):
        gen_imgs.append(add_label(tensor_to_uint8(generated[i]), f"Generated #{i+1}"))
    gen_row = np.hstack(gen_imgs)

    # Separator bar
    sep = np.ones((4, real_row.shape[1]), dtype=np.uint8) * 128

    return np.vstack([real_row, sep, gen_row])


def compute_metrics(
    model, schedule, patches, noise_levels, device,
) -> dict:
    """Compute PSNR between original and denoised patches at each noise level."""
    results = {}
    for t_noise in noise_levels:
        psnrs = []
        for i in range(patches.shape[0]):
            patch = patches[i]
            denoised = denoise_patch(model, schedule, patch, t_noise)

            # PSNR in 16-bit space
            orig_16 = tensor_to_uint16(patch).astype(np.float64)
            dn_16 = tensor_to_uint16(denoised).astype(np.float64)
            mse = np.mean((orig_16 - dn_16) ** 2)
            if mse > 0:
                psnr = 10 * np.log10(65535.0 ** 2 / mse)
            else:
                psnr = float("inf")
            psnrs.append(psnr)
        results[t_noise] = {
            "mean_psnr": np.mean(psnrs),
            "std_psnr": np.std(psnrs),
            "psnrs": psnrs,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Quick train (optional — train a small model for immediate eval)
# ═══════════════════════════════════════════════════════════════════════════

def quick_train(image_path, total_steps, device, patch_size=64, hidden_ch=128):
    """Train a PD model quickly for evaluation purposes."""
    import itertools
    from tqdm import tqdm

    print(f"\n{'='*60}")
    print(f"  Quick-training PD model for {total_steps:,} steps")
    print(f"{'='*60}")

    dataset = SEMPatchDataset(image_path, patch_size=patch_size, mode="random")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=0,
    )

    model = PatchDiffusionCNN(in_channels=1, hidden_channels=hidden_ch).to(device)
    schedule = DDPMSchedule(T=1000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    infinite_loader = itertools.cycle(loader)
    pbar = tqdm(range(total_steps), desc="Quick train", unit="step")

    for step in pbar:
        patches = next(infinite_loader).to(device)
        t = torch.randint(0, schedule.T, (patches.shape[0],), device=device)
        x_t, eps = forward_diffusion(patches, t, schedule)
        v_target = compute_velocity_target(patches, eps, t, schedule)
        v_pred = model(x_t, t)
        loss = F.mse_loss(v_pred, v_target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.5f}")

    # Save quick checkpoint
    ckpt_path = "checkpoints/pd_quick.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "step": total_steps,
        "model": model.state_dict(),
        "schedule_T": schedule.T,
    }, ckpt_path)
    print(f"  Quick model saved to {ckpt_path}")

    return model, schedule


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Patch-Diffusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the 16-bit grayscale image used for training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PD model checkpoint (.pt)")
    parser.add_argument("--quick-train", type=int, default=None, metavar="STEPS",
                        help="Quick-train a model for N steps instead of loading checkpoint")
    parser.add_argument("--output-dir", type=str, default="eval_output",
                        help="Directory to save comparison images")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--hidden-ch", type=int, default=128)
    parser.add_argument("--n-patches", type=int, default=4,
                        help="Number of patches to evaluate")
    parser.add_argument("--noise-levels", type=int, nargs="+",
                        default=[50, 200, 500, 800],
                        help="Noise timesteps for denoising test")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load or train model ───────────────────────────────────────────────
    if args.quick_train:
        model, schedule = quick_train(
            args.input, args.quick_train, device,
            patch_size=args.patch_size, hidden_ch=args.hidden_ch,
        )
    elif args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        T = ckpt.get("schedule_T", 1000)
        schedule = DDPMSchedule(T=T).to(device)
        model = PatchDiffusionCNN(
            in_channels=1, hidden_channels=args.hidden_ch,
        ).to(device)

        # Try loading EMA weights first (better quality)
        if "ema" in ckpt:
            print("  Using EMA weights (recommended)")
            for name, param in model.named_parameters():
                if name in ckpt["ema"]:
                    param.data.copy_(ckpt["ema"][name])
        else:
            model.load_state_dict(ckpt["model"])

        step = ckpt.get("step", "?")
        print(f"  Loaded model at step {step}")
    else:
        print("ERROR: Provide either --checkpoint or --quick-train")
        sys.exit(1)

    model.eval()

    # ── Extract real patches ──────────────────────────────────────────────
    print(f"\nExtracting {args.n_patches} patches from {args.input} ...")
    patches = extract_fixed_patches(
        args.input, args.patch_size, args.n_patches, device=device,
    )
    print(f"  Patches shape: {patches.shape}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: Denoising quality
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Test 1: Denoising Quality")
    print(f"  Noise levels: {args.noise_levels}")
    print(f"{'='*60}")

    denoise_grid = build_denoise_comparison(
        model, schedule, patches, args.noise_levels, device,
    )
    denoise_path = os.path.join(args.output_dir, "denoise_comparison.png")
    cv2.imwrite(denoise_path, denoise_grid)
    print(f"  Saved → {denoise_path}")

    # Compute PSNR metrics
    print("\n  Denoising PSNR (16-bit):")
    metrics = compute_metrics(model, schedule, patches, args.noise_levels, device)
    for t_noise, m in metrics.items():
        print(f"    t={t_noise:>4d}: PSNR = {m['mean_psnr']:.2f} ± {m['std_psnr']:.2f} dB")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: Generation quality
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Test 2: Generation Quality (Real vs Generated)")
    print(f"{'='*60}")

    gen_grid = build_generation_comparison(
        model, schedule, patches, n_generated=args.n_patches,
        patch_size=args.patch_size, device=device,
    )
    gen_path = os.path.join(args.output_dir, "generation_comparison.png")
    cv2.imwrite(gen_path, gen_grid)
    print(f"  Saved → {gen_path}")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Evaluation Complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {args.output_dir}/")
    print(f"    • denoise_comparison.png — Original vs Noised vs Denoised")
    print(f"    • generation_comparison.png — Real patches vs Generated")
    print()
    print("  HOW TO READ THE RESULTS:")
    print("  ─────────────────────────")
    print("  Denoising: Good PD → denoised patches closely match originals")
    print("             Higher noise levels (t=500+) are harder to recover")
    print("  Generation: Good PD → generated patches have similar textures")
    print("              and statistics as real patches from the image")
    print()


if __name__ == "__main__":
    main()
