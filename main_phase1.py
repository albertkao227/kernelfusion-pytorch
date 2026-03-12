#!/usr/bin/env python3
"""
main_phase1.py — Phase 1: Train the Patch-Diffusion (PD) model
================================================================

Trains a velocity-prediction DDPM on random 64×64 crops from a single
16-bit grayscale image.  Every `--sample-every` steps the script runs
a short reverse-diffusion chain and saves a grid of generated patches
so you can visually monitor what the model has learned.

Usage
-----
    # Full training (600 K steps, paper defaults)
    python main_phase1.py --input data/images/lena_gray16.tiff

    # Quick test run
    python main_phase1.py --input data/images/lena_gray16.tiff \\
        --total-steps 500 --batch-size 8 --sample-every 100

    # Run built-in unit tests
    python main_phase1.py --test

Reference: KernelFusion (arXiv:2503.21907), Section 4.1 + Appendix A1.1
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
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
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════


def get_device(requested: str | None = None) -> torch.device:
    """Auto-detect the best available device (CUDA > MPS > CPU)."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── EMA ──────────────────────────────────────────────────────────────────────
class EMAModel:
    """Exponential Moving Average of model weights (decay = 0.9999)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: torch.nn.Module) -> dict:
        """Copy EMA weights into model; return backup of original weights."""
        backup = {}
        for n, p in model.named_parameters():
            backup[n] = p.data.clone()
            p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model: torch.nn.Module, backup: dict):
        for n, p in model.named_parameters():
            p.data.copy_(backup[n])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


# ── Sample generation ────────────────────────────────────────────────────────
@torch.no_grad()
def generate_samples(
    model, schedule, n_samples=4, patch_size=64, sampling_steps=50, device="cpu",
):
    """Run a short reverse-diffusion chain to produce sample patches."""
    model.eval()
    x = torch.randn(n_samples, 1, patch_size, patch_size, device=device)
    ts = torch.linspace(schedule.T - 1, 0, sampling_steps, device=device).long()
    for t_val in ts:
        t_batch = t_val.expand(n_samples)
        v_pred = model(x, t_batch)
        x0_pred = velocity_to_x0(v_pred, x, t_batch, schedule).clamp(-1, 1)
        x = ddpm_reverse_step(x, x0_pred, int(t_val.item()), schedule)
    model.train()
    return x.clamp(-1, 1)


def save_sample_grid(samples: torch.Tensor, path: str, cols: int = 4):
    """Save a batch of [N,1,H,W] patches as a single 16-bit TIFF grid."""
    n = samples.shape[0]
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    h, w = samples.shape[2], samples.shape[3]
    patches = []
    for i in range(n):
        img = samples[i, 0].cpu().numpy()
        img = ((img + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)
        patches.append(img)
    while len(patches) < rows * cols:
        patches.append(np.zeros((h, w), dtype=np.uint16))
    grid_rows = [np.concatenate(patches[r * cols:(r + 1) * cols], axis=1) for r in range(rows)]
    grid = np.concatenate(grid_rows, axis=0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, grid)


# ── Checkpointing ────────────────────────────────────────────────────────────
def save_checkpoint(path, model, ema, optimizer, scheduler, schedule, step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "schedule_T": schedule.T,
        },
        path,
    )


def load_checkpoint(path, model, ema, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Training loop
# ═══════════════════════════════════════════════════════════════════════════


def train_phase1(
    image_path: str,
    total_steps: int = 600_000,
    batch_size: int = 32,
    patch_size: int = 64,
    lr: float = 1e-4,
    T: int = 1000,
    hidden_channels: int = 128,
    device: str | torch.device = "cpu",
    log_every: int = 100,
    save_every: int = 50_000,
    sample_every: int = 10_000,
    output_dir: str = ".",
    resume: str | None = None,
):
    """
    Train the internal Patch-Diffusion model on a single 16-bit image.

    Parameters
    ----------
    image_path      : path to 16-bit grayscale image
    total_steps     : gradient steps (paper: 600 000)
    batch_size      : random crops per step (paper: 32)
    patch_size      : crop size (paper: 64)
    lr              : learning rate (paper: 1e-4)
    T               : DDPM timesteps (paper: 1000)
    hidden_channels : PD model hidden channels (paper: 128)
    device          : cuda / mps / cpu
    log_every       : refresh progress bar every N steps
    save_every      : write checkpoint every N steps
    sample_every    : generate & save sample grid every N steps
    output_dir      : root for checkpoints/ and samples/
    resume          : checkpoint path to resume from

    Returns
    -------
    model, schedule
    """
    device = torch.device(device)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Phase 1 — Patch-Diffusion Training")
    print("=" * 70)
    print(f"  Image        : {image_path}")
    print(f"  Device       : {device}")
    print(f"  Total steps  : {total_steps:,}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Patch size   : {patch_size}×{patch_size}")
    print(f"  LR           : {lr}")
    print(f"  DDPM T       : {T}")
    print(f"  Hidden ch    : {hidden_channels}")
    print(f"  Log every    : {log_every}")
    print(f"  Save every   : {save_every}")
    print(f"  Sample every : {sample_every}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Resume from  : {resume or '(none)'}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    dataset = SEMPatchDataset(image_path, patch_size=patch_size, mode="random")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    print(f"       Image loaded: {dataset.H}×{dataset.W}")
    print(f"       Patch dataset ready (augmented random crops)")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Initialising model …")
    model = PatchDiffusionCNN(in_channels=1, hidden_channels=hidden_channels).to(device)
    schedule = DDPMSchedule(T=T).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       PatchDiffusionCNN — {n_params:,} parameters")
    print(f"       DDPMSchedule — T={T}, β=[{schedule.betas[0]:.1e} → {schedule.betas[-1]:.1e}]")

    # ── Optimiser ─────────────────────────────────────────────────────────
    print("\n[3/4] Setting up optimiser …")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    ema = EMAModel(model, decay=0.9999)
    print(f"       Adam (lr={lr}), CosineAnnealingLR, EMA(0.9999)")
    print(f"       Gradient clipping: max_norm=1.0")

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if resume and os.path.isfile(resume):
        start_step = load_checkpoint(resume, model, ema, optimizer, scheduler, device)
        print(f"\n  ↳ Resumed from step {start_step:,}")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[4/4] Training …\n")
    model.train()
    infinite_loader = itertools.cycle(dataloader)

    pbar = tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        desc="Phase 1",
        unit="step",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    running_loss = 0.0
    loss_count = 0
    t_start = time.time()

    for step in pbar:
        patches = next(infinite_loader).to(device)

        # forward diffusion
        t = torch.randint(0, schedule.T, (patches.shape[0],), device=device)
        x_t, epsilon = forward_diffusion(patches, t, schedule)

        # velocity target & prediction
        v_target = compute_velocity_target(patches, epsilon, t, schedule)
        v_pred = model(x_t, t)

        loss = F.mse_loss(v_pred, v_target)

        # backward + clip + step
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        running_loss += loss.item()
        loss_count += 1

        # ── progress bar update ───────────────────────────────────────────
        if (step + 1) % log_every == 0:
            avg_loss = running_loss / loss_count
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_start
            steps_done = step + 1 - start_step
            speed = steps_done / elapsed if elapsed > 0 else 0
            pbar.set_postfix_str(
                f"loss={avg_loss:.5f}  lr={cur_lr:.2e}  "
                f"grad={grad_norm:.3f}  {speed:.0f} step/s"
            )
            running_loss = 0.0
            loss_count = 0

        # ── checkpoint ────────────────────────────────────────────────────
        if (step + 1) % save_every == 0:
            p = os.path.join(ckpt_dir, f"pd_step{step+1}.pt")
            save_checkpoint(p, model, ema, optimizer, scheduler, schedule, step + 1)
            pbar.write(f"  💾 Checkpoint saved → {p}")

        # ── sample generation (visual progress) ──────────────────────────
        if (step + 1) % sample_every == 0:
            pbar.write(f"\n  🎨 Generating samples at step {step+1} …")

            backup = ema.apply(model)
            samples = generate_samples(
                model, schedule, n_samples=8, patch_size=patch_size, device=device,
            )
            ema.restore(model, backup)

            p = os.path.join(sample_dir, f"samples_step{step+1}.tif")
            save_sample_grid(samples, p, cols=4)
            pbar.write(f"  🎨 Sample grid (8 patches, 2×4) → {p}")

            # Quick stats on the generated samples
            s_min = samples.min().item()
            s_max = samples.max().item()
            s_mean = samples.mean().item()
            s_std = samples.std().item()
            pbar.write(
                f"     Sample stats: min={s_min:.3f} max={s_max:.3f} "
                f"mean={s_mean:.3f} std={s_std:.3f}\n"
            )

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "pd_final.pt")
    save_checkpoint(final_path, model, ema, optimizer, scheduler, schedule, total_steps)
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Phase 1 training complete!")
    print(f"  Total time  : {elapsed/60:.1f} min")
    print(f"  Final ckpt  : {final_path}")
    print(f"  Samples dir : {sample_dir}")
    print(f"{'=' * 70}")

    # Return EMA-smoothed model
    ema.apply(model)
    return model, schedule


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDDPMSchedule(unittest.TestCase):
    """Verify DDPM schedule coefficients."""

    def test_shapes_and_range(self):
        s = DDPMSchedule(T=100)
        self.assertEqual(s.betas.shape, (100,))
        self.assertEqual(s.alpha_bar.shape, (100,))
        self.assertTrue((s.alpha_bar >= 0).all())
        self.assertTrue((s.alpha_bar <= 1).all())
        # α_bar should be monotonically decreasing
        self.assertTrue((s.alpha_bar[1:] <= s.alpha_bar[:-1]).all())

    def test_device_transfer(self):
        s = DDPMSchedule(T=50)
        s.to("cpu")
        self.assertEqual(s.betas.device, torch.device("cpu"))


class TestForwardDiffusion(unittest.TestCase):
    """Verify the forward diffusion process."""

    def test_shape_preservation(self):
        x = torch.randn(4, 1, 64, 64)
        s = DDPMSchedule(T=100)
        t = torch.randint(0, 100, (4,))
        x_t, eps = forward_diffusion(x, t, s)
        self.assertEqual(x_t.shape, x.shape)
        self.assertEqual(eps.shape, x.shape)

    def test_t_zero_close_to_original(self):
        """At t=0, ᾱ₀ ≈ 1, so x_0 should dominate."""
        x = torch.randn(2, 1, 32, 32)
        s = DDPMSchedule(T=100)
        t = torch.zeros(2, dtype=torch.long)
        x_t, _ = forward_diffusion(x, t, s)
        # At t=0, √ᾱ₀ ≈ 0.99995, so x_t ≈ x_0
        diff = (x_t - x).abs().mean()
        self.assertLess(diff.item(), 0.1)


class TestVelocityRoundtrip(unittest.TestCase):
    """v → x₀ → v roundtrip consistency."""

    def test_roundtrip(self):
        x0 = torch.randn(2, 1, 32, 32)
        s = DDPMSchedule(T=100)
        t = torch.randint(1, 100, (2,))
        x_t, eps = forward_diffusion(x0, t, s)
        v = compute_velocity_target(x0, eps, t, s)
        x0_hat = velocity_to_x0(v, x_t, t, s)
        self.assertTrue(torch.allclose(x0, x0_hat, atol=1e-5))


class TestPDModel(unittest.TestCase):
    """Verify PatchDiffusionCNN shapes and forward pass."""

    def test_output_shape(self):
        m = PatchDiffusionCNN(in_channels=1, hidden_channels=64)
        x = torch.randn(2, 1, 64, 64)
        t = torch.randint(0, 100, (2,))
        out = m(x, t)
        self.assertEqual(out.shape, (2, 1, 64, 64))

    def test_different_spatial_sizes(self):
        m = PatchDiffusionCNN(in_channels=1, hidden_channels=32)
        for sz in [32, 48, 64, 128]:
            x = torch.randn(1, 1, sz, sz)
            t = torch.zeros(1, dtype=torch.long)
            out = m(x, t)
            self.assertEqual(out.shape, (1, 1, sz, sz), f"Failed for size {sz}")

    def test_gradient_flow(self):
        m = PatchDiffusionCNN(in_channels=1, hidden_channels=32)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        t = torch.tensor([50])
        out = m(x, t)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


class TestDataset(unittest.TestCase):
    """Test SEMPatchDataset (requires lena_gray16.tiff to exist)."""

    IMAGE_PATH = os.path.join(
        os.path.dirname(__file__), "data", "images", "lena_gray16.tiff"
    )

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__) or ".", "data", "images", "lena_gray16.tiff")),
        "lena_gray16.tiff not found",
    )
    def test_random_mode(self):
        ds = SEMPatchDataset(self.IMAGE_PATH, patch_size=64, mode="random")
        patch = ds[0]
        self.assertEqual(patch.shape, (1, 64, 64))
        self.assertTrue(-1.5 <= patch.mean().item() <= 1.5)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__) or ".", "data", "images", "lena_gray16.tiff")),
        "lena_gray16.tiff not found",
    )
    def test_augmentation_varies(self):
        """Augmentation should produce different patches from the same crop."""
        ds = SEMPatchDataset(self.IMAGE_PATH, patch_size=64, mode="random", augment=True)
        # Draw many samples and check they aren't all identical
        results = [ds[0].sum().item() for _ in range(20)]
        # With flips + rotations, at least some should differ
        unique = len(set(f"{v:.4f}" for v in results))
        self.assertGreater(unique, 1, "Augmentation should produce varied patches")


class TestEMA(unittest.TestCase):
    """Test EMA update and apply/restore."""

    def test_ema_tracks_weights(self):
        m = PatchDiffusionCNN(in_channels=1, hidden_channels=32)
        ema = EMAModel(m, decay=0.0)  # decay=0 → EMA = latest weights
        # Change weights
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(1.0)
        ema.update(m)
        # With decay=0, shadow should equal model
        for n, p in m.named_parameters():
            self.assertTrue(torch.allclose(ema.shadow[n], p.data))

    def test_apply_restore(self):
        m = PatchDiffusionCNN(in_channels=1, hidden_channels=32)
        ema = EMAModel(m, decay=0.5)
        orig = {n: p.data.clone() for n, p in m.named_parameters()}
        backup = ema.apply(m)
        ema.restore(m, backup)
        for n, p in m.named_parameters():
            self.assertTrue(torch.allclose(p.data, orig[n]))


class TestCheckpoint(unittest.TestCase):
    """Test checkpoint save/load roundtrip."""

    def test_save_load_roundtrip(self):
        import tempfile

        m = PatchDiffusionCNN(in_channels=1, hidden_channels=32)
        s = DDPMSchedule(T=50)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = CosineAnnealingLR(opt, T_max=100)
        ema = EMAModel(m)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test_ckpt.pt")
            save_checkpoint(path, m, ema, opt, sched, s, step=42)
            self.assertTrue(os.path.isfile(path))

            # Load into fresh model
            m2 = PatchDiffusionCNN(in_channels=1, hidden_channels=32)
            opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
            sched2 = CosineAnnealingLR(opt2, T_max=100)
            ema2 = EMAModel(m2)

            step = load_checkpoint(path, m2, ema2, opt2, sched2, "cpu")
            self.assertEqual(step, 42)

            # Weights should match
            for (n1, p1), (n2, p2) in zip(
                m.named_parameters(), m2.named_parameters()
            ):
                self.assertTrue(torch.allclose(p1, p2), f"Mismatch in {n1}")


class TestSampleGeneration(unittest.TestCase):
    """Test that sample generation runs and produces valid output."""

    def test_generate(self):
        m = PatchDiffusionCNN(in_channels=1, hidden_channels=32)
        s = DDPMSchedule(T=50)
        samples = generate_samples(m, s, n_samples=2, patch_size=32, sampling_steps=5)
        self.assertEqual(samples.shape, (2, 1, 32, 32))
        self.assertTrue(samples.min() >= -1.0)
        self.assertTrue(samples.max() <= 1.0)

    def test_save_grid(self):
        import tempfile

        samples = torch.randn(4, 1, 32, 32).clamp(-1, 1)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "grid.tif")
            save_sample_grid(samples, p, cols=2)
            self.assertTrue(os.path.isfile(p))
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            self.assertEqual(img.shape, (64, 64))  # 2 rows × 2 cols of 32×32


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args():
    ap = argparse.ArgumentParser(
        description="Phase 1 — Train Patch-Diffusion on a single image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--test", action="store_true", help="Run unit tests and exit")
    ap.add_argument("--input", type=str, default="data/images/lena_gray16.tiff",
                    help="Path to 16-bit grayscale image (default: lena_gray16.tiff)")
    ap.add_argument("--total-steps", type=int, default=600_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=1000,
                    help="DDPM timesteps (default: 1000)")
    ap.add_argument("--hidden-ch", type=int, default=128)
    ap.add_argument("--device", type=str, default=None,
                    help="Device override (cuda/mps/cpu, default: auto)")
    ap.add_argument("--log-every", type=int, default=100,
                    help="Refresh progress bar every N steps")
    ap.add_argument("--save-every", type=int, default=50_000,
                    help="Save checkpoint every N steps")
    ap.add_argument("--sample-every", type=int, default=10_000,
                    help="Generate sample patches every N steps")
    ap.add_argument("--output-dir", type=str, default=".",
                    help="Root dir for checkpoints/ and samples/ folders")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.test:
        # Run unit tests and exit
        print("Running unit tests …\n")
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for cls in [
            TestDDPMSchedule,
            TestForwardDiffusion,
            TestVelocityRoundtrip,
            TestPDModel,
            TestDataset,
            TestEMA,
            TestCheckpoint,
            TestSampleGeneration,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    device = get_device(args.device)

    train_phase1(
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
