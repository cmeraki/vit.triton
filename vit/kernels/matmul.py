import torch
import triton
import triton.language as tl

from .activations import gelu

device = 'cuda:0'
dtype=torch.float16

@triton.autotune(
    configs=[
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 32, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 64, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 64, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, O_ptr,
    A_stride_batch,
    A_stride_height, A_stride_width,
    B_stride_height, B_stride_width,
    O_stride_batch,
    O_stride_height, O_stride_width,
    M, N, K,
    bias_ptr,
    add_bias: tl.constexpr,
    apply_activation: tl.constexpr,
    activation: tl.constexpr,
    bsx: tl.constexpr, bsy: tl.constexpr, bsk: tl.constexpr, group_sz: tl.constexpr
):
    """
    Matrix multiplication by loading rows of A and columns of B to calculate a block of O.
    Uses swizzle to improve L2 cache hit rate, inspired by
    https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

    Is there any performance gains to be done by using Swizzle for batch dimension?
    """
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)

    num_row_programs = tl.num_programs(1)
    num_col_programs = tl.num_programs(2)

    row_idxnew, col_idxnew = tl.swizzle2d(row_idx, col_idx, num_row_programs, num_col_programs, group_sz)

    offset_batch = batch_idx * A_stride_batch

    output = tl.zeros((bsy, bsx), dtype=tl.float32)

    for offset in range(0, K, bsk):
        offset_k = offset + tl.arange(0, bsk)

        # Read offsets from A_ptr
        offset_a = row_idxnew * bsy + tl.arange(0, bsy)
        offset_a = offset_a[:, None]*A_stride_height + offset_k[None, :]*A_stride_width  # bsy * bsk
        mask_a = row_idxnew * bsy + tl.arange(0, bsy)
        mask_a = (mask_a[:, None] < M) & (offset_k[None, :] < K)
        a = tl.load(A_ptr + offset_batch + offset_a, mask_a)

        # Read offset from B_ptr
        offset_b = col_idxnew * bsx + tl.arange(0, bsx)
        offset_b = offset_k[:, None]*B_stride_height + offset_b[None, :]*B_stride_width  # bsk * bsx
        mask_b = col_idxnew * bsx + tl.arange(0, bsx)
        mask_b = (offset_k[:, None] < K) & (mask_b[None, :] < N)
        b = tl.load(B_ptr + offset_b, mask_b)

        output = tl.dot(a, b, output, allow_tf32=True)  # bsy, bsx

    offset_batch_out = batch_idx * O_stride_batch
    offset_or = row_idxnew * bsy + tl.arange(0, bsy)
    offset_oc = col_idxnew * bsx + tl.arange(0, bsx)
    offset_o = offset_or[:, None]*O_stride_height+ offset_oc[None, :]*O_stride_width  # bsy * bsx
    mask_o = (offset_or[:, None] < M) & (offset_oc[None, :] < N)

    if add_bias:
        bias = tl.load(bias_ptr + offset_oc, offset_oc < N)
        output += bias[None, :]

    if apply_activation:
        if activation == 'gelu':
            output = gelu(output)

    tl.store(O_ptr + offset_batch_out + offset_o, output, mask_o)


def matmul_triton(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None, activation: str = None) -> torch.Tensor:
    """
    Implements matrix multiplication between input matrix A and B
    
    Args:
        - A {torch.Tensor}: Input matrix with shape (B, T, Cin) where B is the batch size, T is the sequence length, Cin is the input dimension
        - B {torch.Tensor}: Weight matrix with shape (Cin, Cout) where Cout is the hidden dimension
        - bias {torch.Tensor}: Optionally add a bias to the ouput, shape (1, Cout)
        - activation {str}: Optionally apply activation to the ouput

    Returns:
        - {torch.Tensor}: Output tensor with (B, T, Cout)
    """
    assert len(A.shape) == 3, "First input matrix needs to have 3 dimensions (B, T, C)"
    assert A.device == B.device and A.is_cuda, "Both matrix should be on GPU"

    if bias is not None:
        assert bias.is_cuda, "Bias is not on GPU"
        bias = bias.unsqueeze(0)
        assert bias.shape[1] == B.shape[1], "Bias shape does not match output feature dimension shape"

    if activation:
        assert activation in ["gelu"], f"Only GELU activation supported as of now! Provided: {activation}"

    batch_size, M, K = A.shape
    K, N = B.shape

    grid = lambda meta: (batch_size, triton.cdiv(M, meta["bsy"]), triton.cdiv(N, meta["bsx"]))

    O = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)

    matmul_kernel[grid](
        A, B, O,
        A_stride_batch=A.stride(0),
        A_stride_height=A.stride(1), A_stride_width=A.stride(2),
        B_stride_height=B.stride(0), B_stride_width=B.stride(1),
        O_stride_batch=O.stride(0),
        O_stride_height=O.stride(1), O_stride_width=O.stride(2),
        M=M, N=N, K=K,
        bias_ptr=bias,
        add_bias=True if bias is not None else False,
        activation=activation,
        apply_activation=True if activation else False
    )

    return O


if __name__ == '__main__':
    '''
    python matmul.py -B 512 -M 512 -N 512 -K 512
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument('-B', type=int, default=4)
    parser.add_argument('-M', type=int, default=20)
    parser.add_argument('-K', type=int, default=30)
    parser.add_argument('-N', type=int, default=10)

    args = parser.parse_args()
    print(f'Args: {args}')
    batch_size = args.B
    m = args.M
    k = args.K
    n = args.N

    a = torch.randn((batch_size, m, k), device='cuda', dtype=dtype)
    b = torch.randn((k, n), device='cuda', dtype=dtype)

    bias = torch.randn((n), device='cuda', dtype=dtype)

    y_pytorch = torch.matmul(a, b) + bias
    y_pytorch = torch.nn.functional.gelu(y_pytorch)
    y_triton = matmul_triton(a, b, bias=bias, activation="gelu")

    print(f'Original matrix:\n{a}\n{b}\n{bias}')
    print(f'PyTorch:\n{y_pytorch}')
    print(f'Triton:\n{y_triton}')

    # Unit testing
    assert torch.allclose(y_triton, y_pytorch, atol=1e-1), f"Data does not match, diff: {torch.abs(torch.max(y_pytorch-y_triton))}"

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[64*i for i in range(1, 75)],
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
            args={'batch_size': 1},
        ))
    def benchmark(batch_size, M, N, K, provider):
        quantiles = [0.5, 0.2, 0.8]

        A = torch.randn((batch_size, M, K), device='cuda', dtype=dtype)
        B = torch.randn((K, N), device='cuda', dtype=dtype)
        bias = torch.randn((N), device='cuda', dtype=dtype)

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_triton(A, B, bias, "gelu"), quantiles=quantiles)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.gelu(torch.matmul(A, B) + bias), quantiles=quantiles)

        def gbps(ms): return 2 * batch_size * M * N * K * 1e-12 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(
       show_plots=True,
       print_data=True,
       save_path='./benchmarks/matmul/'
    )
