"""
KernelFusion: Assumption-Free Blind Super-Resolution via Patch Diffusion
(arXiv:2503.21907)

Zero-shot diffusion-based method for 16-bit grayscale SEM super-resolution.

Pipeline:
  Phase 1: Train Patch-Diffusion on the single LR input image
  Phase 2: Reverse diffusion with joint HR + kernel estimation
"""

import argparse
import os
import torch

from core.train import train_internal_diffusion
from core.optimize import optimize_hr_and_kernel
from utils.image_utils import save_sem_16bit, save_kernel


def get_device():
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(
        description="KernelFusion: Zero-Shot SEM Super Resolution"
    )
    
    # Required
    parser.add_argument("--input", type=str, required=True,
                        help="Path to 16-bit grayscale SEM image")
    
    # Output
    parser.add_argument("--output", type=str, default="output_hr.tif",
                        help="Path to save HR output image")
    parser.add_argument("--output-kernel", type=str, default="output_kernel.png",
                        help="Path to save estimated kernel visualization")
    
    # Super-resolution settings
    parser.add_argument("--scale", type=int, default=2,
                        help="Upscaling factor (default: 2)")
    parser.add_argument("--kernel-size", type=int, default=13,
                        help="Kernel size for SR kernel estimation (default: 13)")
    
    # Phase 1 settings
    parser.add_argument("--p1-steps", type=int, default=600_000,
                        help="Phase 1 training steps (paper: 600000)")
    parser.add_argument("--p1-batch-size", type=int, default=32,
                        help="Phase 1 batch size (default: 32)")
    parser.add_argument("--p1-patch-size", type=int, default=64,
                        help="Phase 1 patch size (paper: 64)")
    parser.add_argument("--p1-lr", type=float, default=1e-4,
                        help="Phase 1 learning rate (paper: 1e-4)")
    parser.add_argument("--p1-hidden-ch", type=int, default=128,
                        help="Phase 1 PD model hidden channels (paper: 128)")
    
    # Phase 2 settings
    parser.add_argument("--p2-T-nd", type=int, default=200,
                        help="Phase 2 number of diffusion timesteps (default: 200)")
    parser.add_argument("--p2-n-iter-start", type=int, default=100,
                        help="Phase 2 optimization steps at first timestep (paper: 100)")
    parser.add_argument("--p2-n-iter", type=int, default=20,
                        help="Phase 2 optimization steps per timestep (paper: 20)")
    parser.add_argument("--p2-lr", type=float, default=1e-4,
                        help="Phase 2 learning rate (paper: 1e-4)")
    
    # General
    parser.add_argument("--T", type=int, default=1000,
                        help="DDPM total timesteps (paper: 1000)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cuda/mps/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    # Device selection
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")
    print(f"Input: {args.input}")
    print(f"Scale factor: {args.scale}×")
    print()
    
    # ──────────────────────────────────────────
    # Phase 1: Learn internal patch distribution
    # ──────────────────────────────────────────
    print("=" * 60)
    print("Phase 1: Training Patch-Diffusion on single image")
    print("=" * 60)
    
    trained_pd, schedule = train_internal_diffusion(
        image_path=args.input,
        total_steps=args.p1_steps,
        batch_size=args.p1_batch_size,
        patch_size=args.p1_patch_size,
        lr=args.p1_lr,
        T=args.T,
        hidden_channels=args.p1_hidden_ch,
        device=device,
    )
    
    # ──────────────────────────────────────────
    # Phase 2: Joint HR + Kernel Optimization
    # ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("Phase 2: Reverse diffusion — Joint HR + Kernel estimation")
    print("=" * 60)
    
    hr_tensor, final_kernel = optimize_hr_and_kernel(
        lr_image_path=args.input,
        trained_pd_model=trained_pd,
        schedule=schedule,
        scale_factor=args.scale,
        T_nd=args.p2_T_nd,
        n_iter_start=args.p2_n_iter_start,
        n_iter=args.p2_n_iter,
        lr=args.p2_lr,
        kernel_size=args.kernel_size,
        device=device,
    )
    
    # ──────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────
    print()
    print("Saving results...")
    
    save_sem_16bit(hr_tensor, args.output)
    print(f"  HR image saved to: {args.output}")
    
    save_kernel(final_kernel, args.output_kernel)
    print(f"  Kernel saved to: {args.output_kernel}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()