import torch
import torch.nn as nn
import numpy as np

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega=5.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = np.sqrt(6.0 / in_features) / omega
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class LeakySigmoid(nn.Module):
    def forward(self, x):
        return (1.0 + 1e-4) * torch.sigmoid(x) - 1e-4

class SIRENKernel(nn.Module):
    def __init__(self, hidden_features=256, num_layers=5, omega=5.0, kernel_size=13, channels=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        
        layers = [SirenLayer(2, hidden_features, omega=omega, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_features, hidden_features, omega=omega))
            
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_features, 1)
        self.output_activation = LeakySigmoid()
        
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_features) / omega
            self.output_layer.weight.uniform_(-bound, bound)
            
        self.register_buffer('coord_grid', self._make_coord_grid(kernel_size))
    
    def _make_coord_grid(self, size):
        coords = torch.linspace(-1, 1, size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    
    def forward(self):
        h = self.net(self.coord_grid)
        kernel_values = self.output_activation(self.output_layer(h))
        kernel = kernel_values.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel / (kernel.sum() + 1e-8)
        
        # Expand for depthwise convolution across RGB channels
        return kernel.expand(self.channels, 1, self.kernel_size, self.kernel_size)

    def get_kernel_numpy(self):
        """Get the current kernel as a numpy array for visualization."""
        with torch.no_grad():
            kernel = self.forward()
            # squeeze() removes the batch and channel dimensions so matplotlib can plot it
            return kernel.squeeze().cpu().numpy()