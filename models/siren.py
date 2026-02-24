"""
SIREN-based Implicit Neural Representation (INR) for kernel estimation.

Instead of directly optimizing a discrete kernel tensor, the paper represents
the SR kernel continuously via a SIREN network. This provides implicit
regularization and allows representing kernels at any resolution.

Architecture (from paper Section 4.3 + Appendix A1.3):
  - 5 fully connected layers, 256 nodes each
  - Sinusoidal activations with ω=5 (reduced from SIREN default of 30)
  - Leaky sigmoid output: σ_leaky(x) = (1 + 1e-4) · sigmoid(x) − 1e-4
  - Input: 2D coordinate grid → Output: kernel value at each point
"""

import torch
import torch.nn as nn
import numpy as np


class SirenLayer(nn.Module):
    """A single SIREN layer with sinusoidal activation."""
    
    def __init__(self, in_features, out_features, omega=5.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first
        
        # SIREN initialization (Sitzmann et al., 2020)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = np.sqrt(6.0 / in_features) / omega
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class LeakySigmoid(nn.Module):
    """
    Leaky sigmoid activation for the output layer.
    σ_leaky(x) = (1 + 1e-4) · sigmoid(x) − 1e-4
    
    Allows slight negative values in the kernel (paper Section A1.3).
    """
    
    def forward(self, x):
        return (1.0 + 1e-4) * torch.sigmoid(x) - 1e-4


class SIRENKernel(nn.Module):
    """
    SIREN INR that represents the downscaling SR-kernel continuously.
    
    Input: 2D coordinate grid of shape [K*K, 2] with values in [-1, 1]
    Output: Kernel values [K*K, 1], reshaped to [1, 1, K, K]
    
    The kernel is normalized (sum-to-1) after generation to ensure
    brightness preservation.
    """
    
    def __init__(self, hidden_features=256, num_layers=5, omega=5.0, kernel_size=13):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        # Build SIREN layers
        layers = []
        
        # First layer
        layers.append(SirenLayer(2, hidden_features, omega=omega, is_first=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_features, hidden_features, omega=omega))
        
        # Output layer (linear + leaky sigmoid, no sinusoidal activation)
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_features, 1)
        self.output_activation = LeakySigmoid()
        
        # SIREN init for output layer
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_features) / omega
            self.output_layer.weight.uniform_(-bound, bound)
        
        # Pre-compute coordinate grid
        self.register_buffer('coord_grid', self._make_coord_grid(kernel_size))
    
    def _make_coord_grid(self, size):
        """Create a 2D coordinate grid with values in [-1, 1]."""
        coords = torch.linspace(-1, 1, size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [K*K, 2]
        return grid
    
    def forward(self):
        """
        Generate the SR kernel from the learned INR.
        
        Returns:
            kernel: Normalized kernel [1, 1, K, K]
        """
        # Pass coordinates through SIREN
        h = self.net(self.coord_grid)
        kernel_values = self.output_activation(self.output_layer(h))
        
        # Reshape to 2D kernel
        kernel = kernel_values.view(1, 1, self.kernel_size, self.kernel_size)
        
        # Normalize to sum-to-1 (brightness preservation)
        kernel = kernel / (kernel.sum() + 1e-8)
        
        return kernel
    
    def get_kernel_numpy(self):
        """Get the current kernel as a numpy array for visualization."""
        with torch.no_grad():
            kernel = self.forward()
            return kernel.squeeze().cpu().numpy()
