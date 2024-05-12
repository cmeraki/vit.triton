import torch
import triton
import triton.language as tl

from vit.utils import tensor_info

device = 'cuda:0'

@triton.autotune(
  configs=[
    triton.Config({'bsy': 256, 'bsx': 256}, num_warps=16),
    triton.Config({'bsy': 128, 'bsx': 128}, num_warps=16),
    triton.Config({'bsy': 64, 'bsx': 64}, num_warps=16),
    triton.Config({'bsy': 32, 'bsx': 32}, num_warps=16),
    triton.Config({'bsy': 256, 'bsx': 256}, num_warps=8),
    triton.Config({'bsy': 128, 'bsx': 128}, num_warps=8),
    triton.Config({'bsy': 64, 'bsx': 64}, num_warps=8),
    triton.Config({'bsy': 32, 'bsx': 32}, num_warps=8),
    triton.Config({'bsy': 256, 'bsx': 256}, num_warps=4),
    triton.Config({'bsy': 128, 'bsx': 128}, num_warps=4),
    triton.Config({'bsy': 64, 'bsx': 64}, num_warps=4),
    triton.Config({'bsy': 32, 'bsx': 32}, num_warps=4),
    triton.Config({'bsy': 16, 'bsx': 16}, num_warps=4),
  ],
  key=['batch_size', 'seq_len', 'dim'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, O_ptr,
    A_stride_batch,
    A_stride_height,
    A_stride_width,
    B_stride_height,
    B_stride_width,
    O_stride_batch,
    O_stride_height,
    O_stride_width,
    batch_size,
    seq_len,
    dim: tl.constexpr,
    bsx: tl.constexpr,
    bsy: tl.constexpr,
    bsk: tl.constexpr
):
    """
    Matrix multiplication by loading rows of A
    and columns of B to calculate a block of O.

    This can be further improved by implementing tiling, however
    I am yet to figure out how to use L2 cache in Triton.
    """
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)

    # Batch offset for A and B will be the same
    offset_batch = batch_idx * A_stride_batch
    offset_k = tl.arange(0, dim)

    # Read offsets from A_ptr
    offset_a = row_idx * bsy + tl.arange(0, bsy)
    mask_a = (offset_a[:, None] < seq_len) & (offset_k[None, :] < dim)
    offset_a = offset_a[:, None]*A_stride_height + offset_k[None, :]*A_stride_width  # by * dim
    a = tl.load(A_ptr + offset_batch + offset_a, mask_a)

    # Read offset from B_ptr
    offset_b = col_idx * bsx + tl.arange(0, bsx)
    mask_b = (offset_k[:, None] < dim) & (offset_b[None, :] < seq_len)
    offset_b = offset_k[:, None]*B_stride_height + offset_b[None, :]*B_stride_width  # dim * bx
    b = tl.load(B_ptr + offset_batch + offset_b, mask_b)

    out = tl.dot(a, b, allow_tf32=True)

    offset_out_batch = batch_idx * O_stride_batch
    offset_or = row_idx * bsy + tl.arange(0, bsy)
    offset_oc = col_idx * bsx + tl.arange(0, bsx)
    offset_o = offset_or[:, None]*O_stride_height + offset_oc[None, :]*O_stride_width  # by * bx
    mask_o = (offset_or[:, None] < seq_len) & (offset_oc[None, :] < seq_len)

    tl.store(O_ptr + offset_out_batch + offset_o, out, mask_o)

@tensor_info('matmul3')
def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Implements matrix multiplication between input matrix A and B
    
    Args:
        - A {torch.Tensor}: Input matrix with shape (B, T, Cin) where B is the batch size, T is the sequence length, Cin is the input dimension
        - B {torch.Tensor}: Weight matrix with shape (B, Cin, Cout) where Cout is the hidden dimension

    Returns:
        - {torch.Tensor}: Output tensor with (B, T, Cout)
    """
    assert len(A.shape) == 3, "First input matrix needs to have 3 dimensions (B, T, C)"
    assert len(A.shape) == len(B.shape), "Both matrix should be 3 dimensional"
    assert A.shape[2] == B.shape[1], f"Dimensions are not compatible for matrix multiplication, provided: {A.shape}, {B.shape}"
    assert A.device == B.device and A.is_cuda, "Both matrix should be on GPU"
    assert A.is_contiguous(), "First matrix is not contiguous"
    assert B.is_contiguous(), "Second matrix is not contiguous"

    batch_size, seq_len, dim = A.shape
    dim_out = B.shape[-1]

    grid = lambda meta: (batch_size, triton.cdiv(seq_len, meta["bsy"]), triton.cdiv(dim_out, meta["bsx"]))

    O = torch.empty((batch_size, seq_len, dim_out)).to(A.device, A.dtype)

    matmul_kernel[grid](
        A, B, O,
        A_stride_batch=A.stride(0),
        A_stride_height=A.stride(1),
        A_stride_width=A.stride(2),
        B_stride_height=B.stride(1),
        B_stride_width=B.stride(2),
        O_stride_batch=O.stride(0),
        O_stride_height=O.stride(1),
        O_stride_width=O.stride(2),
        batch_size=batch_size,
        seq_len=seq_len,
        dim=dim,
        bsk=dim
    )

    return O


if __name__ == '__main__':
    '''
    python matmul3.py -batch_size 2 -seq_len 32 -dim 16
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-seq_len', type=int)
    parser.add_argument('-din', type=int)
    parser.add_argument('-dout', type=int)

    args = parser.parse_args()
    print(f'Args: {args}')

    batch_size = args.batch_size
    seq_len = args.seq_len
    din = args.din
    dout = args.dout

    A = torch.randint(0, 5, (batch_size, seq_len, din), device='cuda', dtype=torch.float32)
    B = torch.randint(0, 5, (batch_size, dout, din), device='cuda', dtype=torch.float32)

    B = B.transpose(1, 2).contiguous()
    y_pytorch = torch.matmul(A, B)
    y_triton = matmul_triton(A, B)

    print(f'y_pytorch shape: {y_pytorch.shape}, y_triton shape: {y_triton.shape}')

    print(f'Original matrix:\n{A}\n{B}')
    print(f'PyTorch:\n{y_pytorch}')
    print(f'Triton:\n{y_triton}')

    # Unit testing
    assert torch.allclose(y_triton, y_pytorch), "Data does not match"

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len", "dim"],
            x_vals=[64, 128, 256, 512, 1024],
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
    def benchmark(batch_size, seq_len, dim, provider):
        quantiles = [0.5, 0.2, 0.8]

        x = torch.randint(0, 5, (batch_size, seq_len, dim), device='cuda', dtype=torch.float32)
        y = torch.randint(0, 5, (batch_size, seq_len, dim), device='cuda', dtype=torch.float32)

        y = y.transpose(1, 2).contiguous()

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_triton(x, y), quantiles=quantiles)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(x, y), quantiles=quantiles)

        def gbps(ms): return 2 * seq_len * dim * batch_size * 1e-12 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(
        show_plots=True,
        print_data=True
    )
