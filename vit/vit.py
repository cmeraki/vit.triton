import sys
import math
import torch
import pandas as pd
from torch import nn
from typing import Optional
from loguru import logger

from .load_weights import transfer_pretrained_weights
from .kernels import (
    matmul,
    softmax,
    add,
    matmul3,
    LayerNormTriton,
    Conv2DTriton
)

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="INFO")

device = 'cuda:0'
dtype = torch.float32

class LinearWithBias(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: Optional[str] = None):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = activation

    ##@tensor_info('linear')
    def forward(self, x) -> torch.Tensor:
        return matmul(x, self.weight, self.bias, self.activation)


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: int = 0, #TODO: P1 Add dropout support, currently ViT does not have dropout, so not adding it
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Merging all 3 projections into one
        self.qkv = LinearWithBias(self.d_in, 3*d_out)

    #@tensor_info('self-attn')
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # All three are B x N x 3d_out
        qkv_proj = self.qkv(x)
        # Split back into Q, K, V
        q = qkv_proj[:, :, :self.d_out].contiguous()
        k = qkv_proj[:, :, self.d_out:2*self.d_out]
        v = qkv_proj[:, :, 2*self.d_out:3*self.d_out].contiguous()

        # Inputs are B x N x d_out, B x d_out x N
        # Output is B x N x N
        k = k.transpose(1, 2).contiguous()
        attn_scores = matmul3(q, k, apply_scaling=True, scale_factor=1/math.sqrt(self.d_out))
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

        self.attention = SelfAttention(d_in, d_in)
        self.output = LinearWithBias(self.d_in, self.d_in)

    #@tensor_info('mha')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x N x d_in
        attn_output = self.attention(x)

        return self.output(attn_output)


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

        self.layernorm_before = LayerNormTriton(self.d_in, eps=1e-12)
        self.attention = MultiHeadAttention(self.num_heads, self.d_in, self.d_out)
        self.intermediate = LinearWithBias(self.d_in, 4*self.d_in, activation='gelu')
        self.output = LinearWithBias(4*self.d_in, self.d_in)
        self.layernorm_after = LayerNormTriton(self.d_in, eps=1e-12)

    #@tensor_info('transformer')
    def forward(self, x):
        # B x N x D_out
        attn = self.attention(
            self.layernorm_before(x)
        )

        # First residual connection
        res = add(attn, x)

        out = self.layernorm_after(res)
        out = self.intermediate(out)
        out = self.output(out)

        # Skip connection
        out = add(out, res)

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

    #@tensor_info('encoder')
    def forward(self, x) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x)

        return x


class Embeddings(nn.Module):
    def __init__(self, patch_size, num_patches, patch_dim, hidden_dim):
        super().__init__()

        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros((1, 1, hidden_dim)))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, hidden_dim))
        self.projection = Conv2DTriton(
            in_channels=3,
            out_channels=patch_dim,
            kernel_size=(self.patch_size, self.patch_size)
        )

    #@tensor_info('embedding')
    def forward(self, x) -> torch.Tensor:
        # Input processing
        x = self.projection(x)
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L175
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat([cls_token, x], 1)

        # TODO: P1 Handle brodcast additions in `add`` kernel
        return x + self.position_embeddings


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
        self.layernorm = LayerNormTriton(dim=self.hidden_dim, eps=1e-12)

    def forward(self, x):
        assert x.shape[1:] == (3, self.height, self.width), f"Image size {x.shape[1:]} not matching with the model input size: {3, self.height, self.width}"

        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.layernorm(x)

        return x


if __name__ == '__main__':
    from transformers import ViTConfig, ViTModel

    model_id = 'google/vit-base-patch16-224'
    vit_config = ViTConfig(model_id)

    height, width, channels = vit_config.image_size, vit_config.image_size, vit_config.num_channels
    patch_size = vit_config.patch_size
    hidden_dim = 768
    num_heads = vit_config.num_attention_heads
    num_layers = vit_config.num_hidden_layers

    model: nn.Module = VIT(
        height=height,
        width=width,
        channels=channels,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    model.to(device=device, dtype=dtype)

    pretrained_model = ViTModel.from_pretrained(model_id, add_pooling_layer=False)
    pretrained_model.to(device, dtype)
    pretrained_model.eval()

    model = transfer_pretrained_weights(
        pretrained_model=pretrained_model,
        custom_model=model
    )

    # Torch's way of benchmarking
    from .utils import benchmark
    batch_sizes = [1, 8, 32, 64, 128, 256]
    results = []

    for result in benchmark(pretrained_model, model, batch_sizes=batch_sizes):
        print(f'Batchsize: {result[0]}\tHF Median time: {result[1]}\tTriton Median time: {result[2]}\tRatio: {result[2]/result[1]}')
        results.append(result)

    results_df = pd.DataFrame(results, columns=['Batch Size', 'HF median time', 'Triton median time'])

    results_df.to_csv('./benchmarks/model/benchmark.csv', index=False)

    # Triton's way of benchmarking

    import triton
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['batch_size'],
            x_vals=[1, 2, 4, 8, 16, 24, 32, 48, 64],
            line_arg='provider',
            line_vals=['triton', 'hf'],
            line_names=['Triton', 'Huggingface'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='Time (ms)',
            plot_name='benchmark_vit',
            args={'model1': model, 'model2': pretrained_model}
        )
    )
    def benchmark(batch_size, model1, model2, provider):
        quantiles = [0.5, 0.2, 0.8]
        inp = torch.randn((batch_size, 3, 224, 224), device='cuda', dtype=torch.float32)

        logger.info(f'Benchmarking for batch size: {batch_size} and provider: {provider}')

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: model1(inp), quantiles=quantiles) 
        if provider == 'hf':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: model2(inp), quantiles=quantiles)
        
        return ms, min_ms, max_ms

    benchmark.run(
        # show_plots=True,
        print_data=True,
        save_path='./benchmarks/model/'
    )