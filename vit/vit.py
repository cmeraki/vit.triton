import torch
from torch import nn

from kernels import patching, matmul

device = 'cuda:0'

class VIT(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        patch_size: int,
        hidden_dim: int
    ):
        super().__init__()

        assert height == width, "Height and width should be the same"
        assert height % patch_size == 0, "Height should be divisible by the patch size"
        assert width % patch_size == 0, "Width should be divisible by the patch size"

        self.height = height
        self.width = width
        self.channels = channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        num_patches = (self.height//self.patch_size) * (self.width//self.patch_size)
        patch_dim = self.patch_size*self.patch_size*self.channels

        # Layers initialization
        self.projection = nn.Parameter(torch.randn(patch_dim, self.hidden_dim)).to(device, torch.float32)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, self.hidden_dim)).to(device, torch.float32)
    
    def forward(self, x):
        print(f'Image shape provided: {x.shape}')
        assert x.shape[1:] == (3, self.height, self.width), f"Image size {x.shape[1:]} not matching with the model input size: {3, self.height, self.width}"

        x = patching(x, self.patch_size) 
        print(f'Shape after patching: {x.shape}')
        
        x = matmul(x, self.projection)
        print(f'Shape after matmul: {x.shape}')
        
        x += self.positional_embedding
        print(f'Shape after positional embedding: {x.shape}')

        return x

if __name__ == '__main__':
    from PIL import Image
    import requests
    import numpy as np

    model = VIT(
                height=128,
                width=128,
                channels=3,
                patch_size=4,
                hidden_dim=256
            )

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.resize((128, 128))
    image = torch.Tensor(np.array(image)).to(device='cuda', dtype=torch.float32)

    print(f'Input image shape: {image.shape}')

    out = model(image[None, :].permute(0, 3, 1, 2))
