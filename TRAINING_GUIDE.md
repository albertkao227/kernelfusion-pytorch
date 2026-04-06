# KernelFusion: Training Guide

> Based on the paper: *KernelFusion — Assumption-Free Blind Super-Resolution Using Internal Patch Diffusion*
> ([OpenReview](https://openreview.net/pdf?id=wED9O48qmH))

---

## Overview

KernelFusion is a **zero-shot**, two-phase super-resolution method for 16-bit grayscale images. It requires **no external training data** — only the single low-resolution (LR) input image.

```
┌────────────────────────────────────────────────────────────────────┐
│  Phase 1: Train Patch-Diffusion (PD) model on internal patches    │
│  Phase 2: Reverse diffusion with joint HR + kernel estimation     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `torch>=2.0`, `numpy`, `opencv-python-headless`, `Pillow`, `tqdm`.

### 2. Prepare Your Input Image

The pipeline expects a **16-bit grayscale** image (`.tif` or `.png`). If your image is color, convert it first:

```bash
python data/preprocess.py
```

This converts `data/images/lena_color.tiff` → `data/images/lena_gray16.tiff` (16-bit grayscale).

---

## Phase 1: Train the Patch-Diffusion Model

### Purpose

Learn the **internal patch distribution** of the single LR image. The trained PD model serves as an anti-hallucination prior in Phase 2 — it prevents the SR output from generating textures or structures not present in the original image.

### Architecture — PatchDiffusionCNN

| Property             | Value                |
|----------------------|----------------------|
| Type                 | Pure CNN (no pooling, no attention) |
| Receptive field      | 15×15 pixels         |
| Blocks               | 1 input block (two 3×3 convs) + 6 blocks (3×3 + 1×1 convs) |
| Hidden channels      | 128                  |
| Timestep conditioning| Sinusoidal embedding injected into each block |
| Input / Output       | Single channel (grayscale) |

> **Key design choice:** The restricted 15×15 receptive field forces the model to learn only *local* patch statistics, not global image structure. This prevents overfitting to the single training image.

### Training Loop

```
For 600,000 gradient steps:
    1. Sample a random 64×64 crop from the LR image
    2. Sample random timestep t ∈ {0, ..., T-1}
    3. Forward diffusion: x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
    4. Compute velocity target: v_t = √ᾱ_t · ε − √(1−ᾱ_t) · x_0
    5. PD model predicts velocity: v̂_t = PD(x_t, t)
    6. Loss = MSE(v̂_t, v_t)
    7. Update PD weights via Adam
```

### DDPM Schedule (Shared by Both Phases)

| Parameter  | Value   |
|------------|---------|
| Timesteps T| 1000    |
| β schedule | Linear, β₁ = 1e-4, β_T = 0.02 |
| Parameterization | Velocity prediction (Salimans & Ho, 2022) |

### Hyperparameters

| Parameter        | Paper Default | CLI Flag        |
|------------------|---------------|-----------------|
| Training steps   | 600,000       | `--p1-steps`    |
| Batch size       | 32            | `--p1-batch-size`|
| Patch size       | 64×64         | `--p1-patch-size`|
| Learning rate    | 1e-4          | `--p1-lr`       |
| Hidden channels  | 128           | `--p1-hidden-ch`|
| LR scheduler     | Cosine annealing (η_min = 0.01 × lr) | — |

---

## Phase 2: Joint HR + Kernel Estimation via Reverse Diffusion

### Purpose

Simultaneously estimate the **high-resolution image** and the **degradation kernel** by running reverse diffusion while enforcing LR consistency at every step.

### Components

#### 1. DIP U-Net (Deep Image Prior)

Provides **global structural coherence** that the local PD model cannot capture.

| Property       | Value                                     |
|----------------|-------------------------------------------|
| Architecture   | 5-level encoder-decoder U-Net             |
| Channels       | 32 → 64 → 128 → 256 → 512               |
| Skip connections| Yes, at each level                       |
| Output activation| Tanh (ensures [-1, 1] range)           |
| Trained from   | Scratch during Phase 2                    |

#### 2. SIREN INR (Kernel Estimation)

Represents the SR kernel as a **continuous implicit function** instead of a discrete tensor.

| Property       | Value                                     |
|----------------|-------------------------------------------|
| Architecture   | 5 fully-connected layers, 256 nodes each  |
| Activation     | sin(ω · x), with ω = 5                  |
| Output         | Leaky sigmoid: (1 + 1e-4)·σ(x) − 1e-4   |
| Normalization  | Sum-to-1 (brightness preservation)        |
| Trained from   | Scratch during Phase 2                    |

### Algorithm (Reverse Diffusion Loop)

```
Initialize:
    - Bicubic-upscale the LR image to HR size
    - Noise the bicubic guess to timestep T_nd
    - Initialize DIP U-Net and SIREN INR from scratch
    - Freeze the trained PD model

For t = T_nd, T_nd-1, ..., 1, 0:
    For n_iter optimization steps:
        (a) DIP U-Net refines previous x̂₀ estimate
        (b) Frozen PD model denoises x_t → predicts x̂₀
        (c) DIP U-Net refines PD prediction for global coherence
        (d) SIREN INR generates the SR kernel k
        (e) Downscale: LR_est = (x̂₀ ⊛ k) ↓s
        (f) LR consistency loss = MSE(LR_est, LR_input)
        (g) Backprop through DIP U-Net + SIREN INR only
    DDPM reverse step: x_{t-1} = p(x_{t-1} | x_t, x̂₀)
```

### Hyperparameters

| Parameter            | Paper Default | CLI Flag          |
|----------------------|---------------|-------------------|
| Reverse diffusion steps (T_nd)| 200  | `--p2-T-nd`       |
| Optimization steps (first t) | 100  | `--p2-n-iter-start`|
| Optimization steps (per t)   | 20   | `--p2-n-iter`     |
| Learning rate        | 1e-4          | `--p2-lr`         |
| Kernel size          | 13×13         | `--kernel-size`   |
| Scale factor         | 2×            | `--scale`         |
| LR scheduler         | Cosine annealing (η_min = 0.5 × lr) | — |

---

## Running the Full Pipeline

### Quick Start

```bash
python main.py --input data/images/lena_gray16.tiff
```

### Full Command (All Options)

```bash
python main.py \
    --input data/images/lena_gray16.tiff \
    --output output_hr.tif \
    --output-kernel output_kernel.png \
    --scale 2 \
    --kernel-size 13 \
    --p1-steps 600000 \
    --p1-batch-size 32 \
    --p1-patch-size 64 \
    --p1-lr 1e-4 \
    --p1-hidden-ch 128 \
    --p2-T-nd 200 \
    --p2-n-iter-start 100 \
    --p2-n-iter 20 \
    --p2-lr 1e-4 \
    --T 1000 \
    --device cuda
```

### Outputs

| File                | Description                       |
|---------------------|-----------------------------------|
| `output_hr.tif`     | Super-resolved 16-bit grayscale image |
| `output_kernel.png` | Estimated SR degradation kernel (visualization) |

---

## Project Structure

```
kernelfusion-pytorch/
├── main.py                  # Entry point — runs Phase 1 + Phase 2
├── requirements.txt
├── core/
│   ├── train.py             # Phase 1: PD model training loop
│   ├── optimize.py          # Phase 2: Reverse diffusion + joint optimization
│   └── diffusion.py         # DDPM schedule, forward/reverse diffusion
├── models/
│   ├── unet.py              # PatchDiffusionCNN (Phase 1)
│   ├── dip_unet.py          # DIP U-Net (Phase 2)
│   └── siren.py             # SIREN INR for kernel estimation (Phase 2)
├── data/
│   ├── preprocess.py        # Color → 16-bit grayscale conversion
│   ├── dataset.py           # PatchDataset (random/sliding patches)
│   └── images/              # Input images
└── utils/
    ├── image_utils.py       # 16-bit I/O, patch extraction
    └── physics_utils.py     # Degradation model, kernel utilities
```

---

## Key Equations

| Equation | Formula | Used In |
|----------|---------|---------|
| Forward diffusion | x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε | Phase 1 training |
| Velocity target | v_t = √ᾱ_t · ε − √(1−ᾱ_t) · x₀ | Phase 1 loss |
| Recover x₀ from velocity | x̂₀ = √ᾱ_t · x_t − √(1−ᾱ_t) · v_t | Phase 2 denoising |
| Degradation model | I_LR = (I_HR ⊛ k_s) ↓s | Phase 2 consistency loss |
| Normalization | pixel → [-1, 1] via (val / 32767.5) − 1.0 | Everywhere |

---

## Tips & Notes

1. **GPU strongly recommended.** Phase 1 runs 600K gradient steps; Phase 2 runs ~4,100 optimization steps across 201 diffusion timesteps.
2. **Supported devices:** CUDA > MPS (Apple Silicon) > CPU. Auto-detected by `main.py`.
3. **Image format:** Input must be 16-bit grayscale. Use `data/preprocess.py` to convert color images.
4. **For faster experimentation:** reduce `--p1-steps` (e.g., 50K) and `--p2-T-nd` (e.g., 50) for quicker but lower-quality results.
5. **Kernel size:** The paper uses SIREN INR which can represent kernels at any resolution. Default 13×13 works well for 2× SR.
