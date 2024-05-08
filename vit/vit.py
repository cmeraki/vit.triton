import torch
from torch import nn

from vit.utils import tensor_info
from vit.kernels import (
    patching,
    matmul,
    softmax,
    layernorm,
    add
)

from loguru import logger

device = 'cuda:0'
dtype = torch.float32

# TODO: Add activation support

class SelfAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: int = 0, #TODO: Add dropout support
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Initializing Q, K, V with shapes (d_in, d_out)
        self.q_proj = nn.Parameter(torch.randn(self.d_in, self.d_out)).to(device, dtype)
        self.k_proj = nn.Parameter(torch.randn(self.d_in, self.d_out)).to(device, dtype)
        self.v_proj = nn.Parameter(torch.randn(self.d_in, self.d_out)).to(device, dtype)

    @tensor_info('self-attn')
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # All three are B x N x d_out
        # TODO: Possible to merge all these 3 matmuls in single kernel?
        q = matmul(x, self.q_proj)
        k = matmul(x, self.k_proj)
        v = matmul(x, self.v_proj)

        # Inputs are B x N x d_out, B x N x d_out
        # Output is B x N x N
        attn_scores = matmul(q, k.T)

        # TODO: Fuse matmul and sqrt
        attn_scores = attn_scores/torch.sqrt(self.d_out)
        attn_scores = softmax(attn_scores)

        # Inputs are B x N x N, B x N x d_out
        # Output is B x N x d_out
        context_vec = matmul(attn_scores, v)

        return context_vec

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_in: int,
        d_out: int,
    ):
        super().__init__()

        assert d_in%num_heads == 0, f'Input dimension should be equally divided amongst all heads. d_in%num_heads needs to be 0. Current: {d_in%num_heads}'
        assert d_in/num_heads == d_out, f'`d_out` is not equal to `d_in/num_heads`. Current: {d_in/num_heads}, {d_out}'

        self.num_heads = num_heads
        self.d_in = d_in
        self.d_out = d_out
        self.o_proj = nn.Parameter(torch.randn(self.d_in, self.d_in)).to(device, dtype)

        self.layers = []
        for _ in range(self.num_heads):
            self.layers.append(SelfAttention(d_in=d_in, d_out=d_out))


    @tensor_info('mha')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.num_heads):
            # Naive: Process one head at a time
            # Each elem in output will be B x N x d_out
            # TODO: Implement MHA in a more optimized kernel
            outputs.append(
                self.layers[i](x)
            )

        # B x N x d_in
        out = torch.cat(
            outputs,
            dim=-1
        ).contiguous()

        out = matmul(out, self.o_proj)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_in: int,
        d_out: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_in = d_in
        self.d_out = d_out


        self.mha = MultiHeadAttention(self.num_heads, self.d_in, self.d_out)
        self.ffn_1 = nn.Parameter(torch.randn(self.d_in, 4*self.d_in)).to(device, dtype)
        self.ffn_2 = nn.Parameter(torch.randn(4*self.d_in, self.d_in)).to(device, dtype)

    def forward(self, x):
        # B x N x D_out
        attn = self.mha(x)

        # Skip connection
        intermediate = add(attn, x)
        intermediate = layernorm(intermediate)

        # B x N x 4D_out
        out = matmul(intermediate, self.ffn_1)
        # B x N x D_out
        out = matmul(out, self.ffn_2)

        # Skip connection
        out = add(out, intermediate)

        return out


class VIT(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        patch_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int
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
        self.num_heads = num_heads
        self.num_layers = num_layers

        assert self.hidden_dim % self.num_heads == 0, f"Hidden dimension should be divisible by number of heads, provided: {self.hidden_dim} {self.num_heads}"

        num_patches = (self.height//self.patch_size) * (self.width//self.patch_size)
        patch_dim = self.patch_size*self.patch_size*self.channels
        d_out = int(self.hidden_dim/self.num_heads)


        # Layers initialization
        self.projection = nn.Parameter(torch.randn(patch_dim, self.hidden_dim)).to(device, dtype)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, self.hidden_dim)).to(device, dtype)
        self.transformer_blocks = [Transformer(num_heads=self.num_heads, d_in=self.hidden_dim, d_out=d_out) for _ in range(self.num_layers)]

    def forward(self, x):
        print(f'Image shape provided: {x.shape}')
        assert x.shape[1:] == (3, self.height, self.width), f"Image size {x.shape[1:]} not matching with the model input size: {3, self.height, self.width}"


        # Input processing
        x = patching(x, self.patch_size)
        # TODO: Possible to fuse kernels?
        x = matmul(x, self.projection)
        x = add(x, self.positional_embedding)

        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x)

        return x

if __name__ == '__main__':
    from PIL import Image
    import requests
    import numpy as np

    height, width = 128, 128

    model = VIT(
        height=height,
        width=width,
        channels=3,
        patch_size=16,
        hidden_dim=256,
        num_heads=1,
        num_layers=1
    )

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.resize((height, width))
    image = torch.Tensor(np.array(image)).to(device=device, dtype=dtype)

    print(f'Input image shape: {image.shape}')

    out = model(image[None, :].permute(0, 3, 1, 2))
