# kernelfusion-pytorch
Pytorch implementation for KernelFusion arXiv:2503.21907 

KernelFusion is a zero-shot, diffusion-based approach for blind super-resolution (Blind-SR). It breaks away from standard predefined kernel assumptions (like simple Gaussian downscaling) by learning an image-specific patch prior and jointly estimating both an unrestricted super-resolution kernel and the high-resolution (HR) reconstruction.

## Architecture Overview
The pipeline leverages cross-scale patch consistency and consists of two sequential phases:

**Phase 1 (Patch Distribution Learning)**: Trains a localized Patch Diffusion (PD) model exclusively on the low-resolution (LR) input image. This step maps and learns the internal patch distribution of the specific image without relying on external datasets.

**Phase 2 (Blind SR & Kernel Estimation)**: Performs joint optimization. The trained PD model acts as a prior to guide the HR estimation toward the correct patch distribution. Simultaneously, a refinement U-Net and a SIREN (Sinusoidal Representation Network) implicit kernel representation are trained under a consistency loss to ensure that convolving the predicted HR image with the estimated kernel perfectly reproduces the original LR input.

## Usage
The process operates strictly on a per-image, zero-shot basis. Follow the steps below in order to process a target low-resolution image.

### Phase 1: Patch Diffusion
1. Train the Patch Diffusion Model 
Extracts patches from the target image and trains the diffusion prior.

```
python phase1_train_patch_diffusion.py \
    --input_image ./data/lr_image.png \
    --output_dir ./checkpoints/phase1/ \
    --num_steps 5000 \
    --batch_size 16
```

2. Evaluate the Patch Diffusion Model (Optional but recommended) 
Verifies that the diffusion model has successfully learned the internal patch statistics before moving to the heavier Phase 2 optimization.

```
python phase1_evaluate_patch_diffusion.py \
    --input_image ./data/lr_image.png \
    --checkpoint ./checkpoints/phase1/patch_diffusion.pth \
    --results_dir ./results/phase1_eval/
```

### Phase 2: Joint Super-Resolution and Kernel Estimation
3. Run the Super-Resolution Pipeline 
This script initializes the U-Net and SIREN components, leveraging the pre-trained Phase 1 weights to enforce patch consistency while recovering the unrestricted degradation kernel.

```
python phase2_superresolution.py \
    --input_image ./data/lr_image.png \
    --pd_checkpoint ./checkpoints/phase1/patch_diffusion.pth \
    --scale_factor 4 \
    --output_dir ./results/phase2_final/
```