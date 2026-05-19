# kernelfusion-pytorch
Pytorch implementation for KernelFusion arXiv:2503.21907 

KernelFusion is a zero-shot, diffusion-based approach for blind super-resolution (Blind-SR). It breaks away from standard predefined kernel assumptions (like simple Gaussian downscaling) by learning an image-specific patch prior and jointly estimating both an unrestricted super-resolution kernel and the high-resolution (HR) reconstruction.

## Architecture Overview
The pipeline leverages cross-scale patch consistency and consists of two sequential phases:

**Phase 1 (Patch Distribution Learning)**: Trains a localized Patch Diffusion (PD) model exclusively on the low-resolution (LR) input image. This step maps and learns the internal patch distribution of the specific image without relying on external datasets.

**Phase 2 (Blind SR & Kernel Estimation)**: Performs joint optimization. The trained PD model acts as a prior to guide the HR estimation toward the correct patch distribution. Simultaneously, a refinement U-Net and a SIREN (Sinusoidal Representation Network) implicit kernel representation are trained under a consistency loss to ensure that convolving the predicted HR image with the estimated kernel perfectly reproduces the original LR input.

## Usage

The pipeline operates strictly on a per-image, zero-shot basis. Parameters (like input image paths and checkpoints) are set directly inside the `__main__` blocks of each script. 

To process an image, edit the file paths in the scripts as needed, and then execute them in the following order:

```bash
# 1. Train the image-specific patch diffusion prior
python phase1_train_patch_diffusion.py

# 2. (Optional) Evaluate the learned patch statistics
python phase1_evaluate_patch_diffusion.py

# 3. Run the joint U-Net + SIREN optimization for super-resolution
python phase2_superresolution.py
