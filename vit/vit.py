import math
import torch
from torch import nn

from .utils import tensor_info, transfer_pretrained_weights
from .kernels import (
    patching,
    matmul,
    softmax,
    add,
    matmul3,
    LayerNormTriton,
)

from loguru import logger

device = 'cuda:0'
dtype = torch.float32

# TODO: Fuse matmul and bias
# TODO: Add activation support

class LinearWithBias(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    @tensor_info('linear')
    def forward(self, x) -> torch.Tensor:
        x = matmul(x, self.weight)
        # TODO: Handle broadcast addition
        x = torch.add(x, self.bias)

        return x


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
        self.query = LinearWithBias(self.d_in, self.d_out)
        self.key = LinearWithBias(self.d_in, self.d_out)
        self.value = LinearWithBias(self.d_in, self.d_out)

    @tensor_info('self-attn')
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # All three are B x N x d_out
        # TODO: Possible to merge all these 3 matmuls in single kernel?
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Inputs are B x N x d_out, B x N x d_out
        # Output is B x N x N
        k = k.transpose(1, 2).contiguous()
        attn_scores = matmul3(q, k)

        # TODO: Fuse matmul and sqrt
        attn_scores = attn_scores/math.sqrt(self.d_out)
        attn_scores = softmax(attn_scores)

        # Inputs are B x N x N, B x N x d_out
        # Output is B x N x d_out
        context_vec = matmul3(attn_scores, v)

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

        self.attention = nn.ModuleList([SelfAttention(d_in=d_in, d_out=d_out) for _ in range(self.num_heads)])
        self.output = LinearWithBias(self.d_in, self.d_in)

    @tensor_info('mha')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for attn in self.attention:
            # Naive: Process one head at a time
            # Each elem in output will be B x N x d_out
            # TODO: Implement MHA in a more optimized kernel
            outputs.append(
                attn(x)
            )

        # B x N x d_in
        out = torch.cat(
            outputs,
            dim=-1
        ).contiguous()

        out = self.output(out)

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

        self.attention = MultiHeadAttention(self.num_heads, self.d_in, self.d_out)
        self.intermediate = LinearWithBias(self.d_in, 4*self.d_in)
        self.output = LinearWithBias(4*self.d_in, self.d_in)

        self.layernorm_before = LayerNormTriton(self.d_in, eps=1e-12)
        self.layernorm_after = LayerNormTriton(self.d_in, eps=1e-12)

    @tensor_info('transformer')
    def forward(self, x):
        # B x N x D_out
        attn = self.attention(x)

        # Skip connection
        res = add(attn, x)
        res = self.layernorm_before(res)

        # B x N x 4D_out
        out = self.intermediate(res)
        # B x N x D_out
        out = self.output(out)

        # Skip connection
        out = add(out, res)
        out = self.layernorm_after(out)

        return out

class Encoder(nn.Module):
    def __init__(self, num_layers:int, num_heads: int, hidden_dim: int, d_out: int):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.d_out = d_out

        self.layer = nn.ModuleList(
            Transformer(num_heads=self.num_heads, d_in=self.hidden_dim, d_out=d_out) for _ in range(self.num_layers)
        )

    @tensor_info('encoder')
    def forward(self, x) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x)

        return x


class Embeddings(nn.Module):
    def __init__(self, patch_size, num_patches, patch_dim, hidden_dim):
        super().__init__()


        self.patch_size = patch_size

        self.cls_token = torch.Tensor((1, 1, hidden_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, hidden_dim))
        self.projection = LinearWithBias(patch_dim, hidden_dim)

    @tensor_info('embedding')
    def forward(self, x) -> torch.Tensor:
        # Input processing
        x = patching(x, self.patch_size)

        # TODO: Possible to fuse kernels?
        x = self.projection(x)
        x = torch.cat([x, self.cls_token])
        x = add(x, self.position_embeddings)

        return x

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
        self.embeddings = Embeddings(patch_size=self.patch_size, num_patches=num_patches, patch_dim=patch_dim, hidden_dim=self.hidden_dim)
        self.encoder = Encoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=self.hidden_dim, d_out=d_out)


    def forward(self, x):
        print(f'Image shape provided: {x.shape}')
        assert x.shape[1:] == (3, self.height, self.width), f"Image size {x.shape[1:]} not matching with the model input size: {3, self.height, self.width}"

        x = self.embeddings(x)
        x = self.encoder(x)

        return x


if __name__ == '__main__':
    from PIL import Image
    import requests
    import numpy as np

    height, width = 224, 224

    model = VIT(
        height=height,
        width=width,
        channels=3,
        patch_size=16,
        hidden_dim=768,
        num_heads=12,
        num_layers=12
    )
    model.to(device, dtype)

    model = transfer_pretrained_weights(
        model_id='google/vit-base-patch16-224',
        custom_model=model
    )


    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.resize((height, width))
    image = torch.Tensor(np.array(image)).to(device=device, dtype=dtype)

    print(f'Input image shape: {image.shape}')

    out = model(image[None, :].permute(0, 3, 1, 2))
