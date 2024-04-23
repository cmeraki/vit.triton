import torch
from torch import nn

from kernels import patching, matmul

class VIT(nn.Module):
    def __init__(
        model_id: str,
        height: int,
        width: int,
        patch_size: int
    ):
        super().__init__()
        assert height == width, "Height and width should be the same"
        assert height % patch_size == 0, "Height should be divisible by the patch size"
        assert width % patch_size == 0, "Width should be divisible by the patch size"

        self.height = height
        self.widht = width
        self.patch_size = patch_size
        
        # Layers initialization
        self.patching = patching
        self.positional_embedding = None
    
    def forward(self, x):
        assert x.shape == (self.height, self.weight)
        x = self.patching(x, self.patch_size, self.positional_embedding)
        x += self.positional_embedding

        return x

