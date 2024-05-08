import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.autotune(
  configs=[
    triton.Config({'bsy': 128, 'bsx': 256}, num_warps=8),
    triton.Config({'bsy': 64, 'bsx': 256}, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 128}, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 64}, num_warps=4),
    triton.Config({'bsy': 64, 'bsx': 128}, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 32}, num_warps=4),
    triton.Config({'bsy': 64, 'bsx': 32}, num_warps=2),
    triton.Config({'bsy': 32, 'bsx': 64}, num_warps=2),
    triton.Config({'bsy': 128, 'bsx': 256}, num_warps=8),
    triton.Config({'bsy': 256, 'bsx': 128}, num_warps=8),
    triton.Config({'bsy': 256, 'bsx': 64,}, num_warps=4),
    triton.Config({'bsy': 64, 'bsx': 256}, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 128}, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 64}, num_warps=4),
    triton.Config({'bsy': 64, 'bsx': 128}, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 32}, num_warps=4),
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
    M, N, K: tl.constexpr,
    bsx: tl.constexpr, bsy: tl.constexpr
):
    """
    Matrix multiplication by loading rows of A
    and columns of B to calculate a block of O.

    This can be further improved by implementing tiling, however
    I am yet to figure out how to use L2 cache in Triton.
    """
    batch_idx = tl.program_id(axis=0)
    # Load apt data into memory
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)

    offset_batch = batch_idx * A_stride_batch
    offset_k = tl.arange(0, K)

    # Read offsets from A_ptr
    offset_a = row_idx * bsy + tl.arange(0, bsy)
    offset_a = offset_a[:, None]*A_stride_height + offset_k[None, :]*A_stride_width  # by * K
    mask_a = row_idx * bsy + tl.arange(0, bsy)
    mask_a = (mask_a[:, None] < M) & (offset_k[None, :] < K)
    a = tl.load(A_ptr + offset_batch + offset_a, mask_a)

    # Read offset from B_ptr
    offset_b = col_idx * bsx + tl.arange(0, bsx)
    offset_b = offset_k[:, None]*B_stride_height + offset_b[None, :]*B_stride_width  # K * bx
    mask_b = col_idx * bsx + tl.arange(0, bsx)
    mask_b = (offset_k[:, None] < K) & (mask_b[None, :] < N)
    b = tl.load(B_ptr + offset_b, mask_b)

    o = tl.dot(a, b)

    offset_batch_out = batch_idx * O_stride_batch
    offset_or = row_idx * bsy + tl.arange(0, bsy)
    offset_oc = col_idx * bsx + tl.arange(0, bsx)
    offset_o = offset_or[:, None]*O_stride_height+ offset_oc[None, :]*O_stride_width  # by * bx
    mask_o = (offset_or[:, None] < M) & (offset_oc[None, :] < N)

    tl.store(O_ptr + offset_batch_out + offset_o, o, mask_o)

def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Implements matrix multiplication between input matrix A and B
    
    Args:
        - A {torch.Tensor}: Input matrix with shape (B, T, Cin) where B is the batch size, T is the sequence length, Cin is the input dimension
        - B {torch.Tensor}: Weight matrix with shape (Cin, Cout) where Cout is the hidden dimension

    Returns:
        - {torch.Tensor}: Output tensor with (B, T, Cout)

    Output will be (B, T, T)
    """
    assert len(A.shape) == 3, "First input matrix needs to have 3 dimensions (B, T, C)"
    assert A.device == B.device and A.is_cuda, "Both matrix should be on GPU"

    batch_size, M, K = A.shape
    K, N = B.shape

    grid = lambda meta: (batch_size, triton.cdiv(M, meta["bsy"]), triton.cdiv(N, meta["bsx"]))

    O = torch.empty((batch_size, M, N)).to(A.device, A.dtype)

    matmul_kernel[grid](
        A, B, O,
        A_stride_batch=A.stride(0),
        A_stride_height=A.stride(1), A_stride_width=A.stride(2),
        B_stride_height=B.stride(0), B_stride_width=B.stride(1),
        O_stride_batch=O.stride(0),
        O_stride_height=O.stride(1), O_stride_width=O.stride(2),
        M=M, N=N, K=K#, bsx=bsx, bsy=bsy
    )

    return O


if __name__ == '__main__':
    '''
    python matmul.py -M 512 -N 512 -K 512
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument('-B', type=int)
    parser.add_argument('-M', type=int)
    parser.add_argument('-K', type=int)
    parser.add_argument('-N', type=int)

    args = parser.parse_args()
    print(f'Args: {args}')
    batch_size = args.B
    M = args.M
    K = args.K
    N = args.N

    A = torch.randint(0, 10, (batch_size, M, K), device='cuda', dtype=torch.float32)
    B = torch.randint(0, 5, (K, N), device='cuda', dtype=torch.float32)

    assert A.shape[2] == B.shape[0], 'Matrix are not compatible for multiplication'

    y_pytorch = torch.matmul(A, B)
    y_triton = matmul_triton(A, B)

    print(f'Original matrix:\n{A}\n{B}')
    print(f'PyTorch:\n{y_pytorch}')
    print(f'Triton:\n{y_triton}')

    # Unit testing
    assert torch.allclose(y_triton, y_pytorch), "Data does not match"

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N"],
            x_vals=[64, 128, 256, 512, 1024, 2056, 4096],
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
            args={'batch_size': 1, 'K': 32},
        ))
    def benchmark(batch_size, M, N, K, provider):
        quantiles = [0.5, 0.2, 0.8]

        A = torch.randint(0, 10, (batch_size, M, K), device='cuda', dtype=torch.float32)
        B = torch.randint(0, 5, (K, N), device='cuda', dtype=torch.float32)

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_triton(A, B), quantiles=quantiles)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)

        def gbps(ms): return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(
        show_plots=True,
        print_data=True
    )
