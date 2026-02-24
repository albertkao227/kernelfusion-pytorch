"""
Dataset for extracting patches from a single SEM image.

Supports two modes:
  - 'random': Random 64×64 crops (used for Phase 1 training, paper default)
  - 'sliding': Sliding window patches (alternative for exhaustive coverage)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.image_utils import load_sem_16bit
import numpy as np


class SEMPatchDataset(Dataset):
    """
    Dataset that provides patches from a single 16-bit SEM image.
    
    In 'random' mode (paper default), each __getitem__ call returns a 
    randomly cropped patch. The dataset length is set artificially high
    to enable long training runs without epoch boundaries.
    
    In 'sliding' mode, patches are pre-extracted with a sliding window.
    """
    
    def __init__(self, image_path, patch_size=64, stride=16, mode='random', 
                 virtual_size=10000):
        """
        Args:
            image_path: Path to 16-bit grayscale SEM image
            patch_size: Size of patches to extract (paper: 64)
            stride: Stride for sliding window mode
            mode: 'random' (paper default) or 'sliding'
            virtual_size: Virtual dataset size for random mode
        """
        self.patch_size = patch_size
        self.mode = mode
        self.virtual_size = virtual_size
        
        # Load the full image
        self.img_tensor = load_sem_16bit(image_path, device='cpu')  # [1, 1, H, W]
        self.H = self.img_tensor.shape[2]
        self.W = self.img_tensor.shape[3]
        
        if self.H < patch_size or self.W < patch_size:
            raise ValueError(
                f"Image size ({self.H}×{self.W}) is smaller than "
                f"patch size ({patch_size}×{patch_size})"
            )
        
        if mode == 'sliding':
            # Pre-extract sliding window patches
            self.patches = F.unfold(
                self.img_tensor, kernel_size=patch_size, stride=stride
            )
            # Reshape: [1, C*P*P, N] → [N, 1, P, P]
            self.patches = self.patches.view(
                1, 1, patch_size, patch_size, -1
            ).squeeze(0).permute(3, 0, 1, 2)
    
    def __len__(self):
        if self.mode == 'random':
            return self.virtual_size
        else:
            return self.patches.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == 'random':
            # Random crop from the full image
            y = np.random.randint(0, self.H - self.patch_size + 1)
            x = np.random.randint(0, self.W - self.patch_size + 1)
            patch = self.img_tensor[0, :, y:y+self.patch_size, x:x+self.patch_size]
            return patch  # [1, patch_size, patch_size]
        else:
            return self.patches[idx]  # [1, patch_size, patch_size]