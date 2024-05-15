import torch
import triton
import triton.language as tl

device = 'cuda'
dtype = torch.float32

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 16}, num_warps=4),
    ],
    key=['num_rows', 'num_cols']
)
@triton.jit
def layernorm_kernel(
    a_ptr,
    batch_stride,
    row_stride,
    col_stride,
    num_rows,
    num_cols,  # Number of columns
    weight_ptr,
    bias_ptr,
    eps,
    out_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """
    IDEA 1: Merge batch and seq len dimension into 1
    IDEA 2: Use tiled row approach
    """
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)

    batch_offset = batch_idx * batch_stride
    row_offset = row_idx * row_stride

    local_sum = 0.0
    for offset in range(0, num_cols, BLOCK_SIZE):
        local_offset = batch_offset + row_offset + offset + tl.arange(0, BLOCK_SIZE)
        mask = offset + tl.arange(0, BLOCK_SIZE) < num_cols
        data = tl.load(a_ptr + local_offset, mask=mask, other=0.0)

        local_sum += tl.sum(data)

    mean = local_sum/num_cols

    local_std = 0.0
    for offset in range(0, num_cols, BLOCK_SIZE):
        local_offset = batch_offset + row_offset + offset + tl.arange(0, BLOCK_SIZE)
        mask = offset + tl.arange(0, BLOCK_SIZE) < num_cols
        data = tl.load(a_ptr + local_offset, mask=mask, other=mean)

        x = data-mean
        x = x*x

        local_std += tl.sum(x)

    std = local_std / num_cols + eps
    std = tl.sqrt(std)

    for offset in range(0, num_cols, BLOCK_SIZE):
        local_offset = offset + tl.arange(0, BLOCK_SIZE)
        mask = local_offset < num_cols
        w = tl.load(weight_ptr + local_offset, mask=mask, other=0.0)
        b = tl.load(bias_ptr + local_offset, mask=mask, other=0.0)

        local_offset += row_offset + batch_offset
        mask = offset + tl.arange(0, BLOCK_SIZE) < num_cols
        x = tl.load(a_ptr + local_offset, mask=mask, other=0.0)

        norm = w*((x-mean)/std) + b

        tl.store(out_ptr+local_offset, norm, mask=mask)


def layernorm_triton(A: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Performs layernorm on the input matrix

    Args:
        A (torch.Tensor): Input matrix of size B * N * D where B is the batch size, N is the sequence length and D is the embedding dimension
        weight (torch.Tensor): Weight matrix of size (D, 1)
        bias (torch.Tensor): Bias matrix (D, 1)
        eps (float): Epsilon value in layer norm for smoothening

    Returns:
        {torch.Tensor}: Same size as input matrix
    """

    assert A.is_contiguous(), 'Matrix is not contiguous'
    assert A.is_cuda, 'Matrix is not on GPU'
    assert len(A.shape) == 3, f"Only 3 dimensional matrix is supported as input"

    # Output tensor
    O = torch.empty_like(A)

    batches, seq_len, dim = A.shape
    grid = (batches, seq_len, )

    layernorm_kernel[grid](
        a_ptr=A,
        batch_stride=A.stride(0),
        row_stride=A.stride(1),
        col_stride=A.stride(2),
        num_rows=seq_len,
        num_cols=dim,
        weight_ptr=weight,
        bias_ptr=bias,
        eps=eps,
        out_ptr=O
    )

    return O

class LayerNormTriton(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()

        self.dim = dim
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(self.dim))
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm_triton(
            x, self.weight, self.bias, self.eps
        )

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-B', type=int)
    parser.add_argument('-N', type=int)
    parser.add_argument('-D', type=int)

    args = parser.parse_args()

    B = args.B
    N = args.N
    D = args.D

    a = torch.randn((B, N, D), device=device, dtype=dtype)
    _shape = (a.shape[-1], )
    weight = torch.ones(_shape, device=device, dtype=dtype)
    bias = torch.zeros(_shape, device=device, dtype=dtype)
    eps = 1e-12

    y_pytorch = torch.nn.functional.layer_norm(a, _shape, weight, bias, eps).to(dtype)
    y_triton = layernorm_triton(a, weight=weight, bias=bias, eps=eps)

    print(f'Original tensor\n{a}')
    print(f'PyTorch layer norm\n{y_pytorch}')
    print(f'Triton layer norm\n{y_triton}')

    assert torch.allclose(y_triton, y_pytorch, atol=1e-6), f'Data does not match, diff = {torch.max(torch.abs(y_pytorch-y_triton))}'

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'], # argument names to use as an x-axis for the plot
            # different possible values for `x_name`
            x_vals=[128*i for i in range(2, 50, 2)],
            # argument name whose value corresponds to a different line in the plot
            line_arg='provider',
            line_vals=[
                'triton',
                'torch',
            ],
            line_names=[
                "Triton",
                "Torch (native)",
            ],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name="Performance",
            # values for function arguments not in `x_names` and `y_name`
            args={'B': 1, 'D': 768},
        ))
    def benchmark(B, N, D, provider):
        quantiles = [0.5, 0.2, 0.8]

        a = torch.randint(0, 10, (B, N, D), device=device, dtype=dtype)
        _shape = (a.shape[-1], )
        weight = torch.randn(_shape, device=device, dtype=dtype)
        bias = torch.randn(_shape, device=device, dtype=dtype)
        eps = 1e-5

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: layernorm_triton(a, weight=weight, bias=bias, eps=eps), quantiles=quantiles)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(a, _shape, weight, bias, eps).to(dtype), quantiles=quantiles)

        def gbps(ms): return 2 * a.nelement() * a.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(
        show_plots=True,
        print_data=True
    )
