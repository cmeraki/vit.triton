import math
import torch
import triton
import triton.language as tl

device = 'cuda:0'
dtype = torch.float32

@triton.autotune(
    configs=[
        # triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 8}, num_stages=3, num_warps=8),
        # triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 64, 'bsx': 32, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        # triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        # triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8),
        # triton.Config({'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8),
        # triton.Config({'bsy': 256, 'bsx': 64, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 64, 'group_sz': 8}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 4}, num_stages=3, num_warps=8),
        # triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=5, num_warps=2),
        # triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8),
        # triton.Config({'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8),
        # triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 4}, num_stages=4, num_warps=4),
        # triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4)
    ],
    key=['batch_size', 'seq_len', 'dim', 'dim_out'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, O_ptr,
    A_stride_batch,
    A_stride_height,
    A_stride_width,
    B_stride_batch,
    B_stride_height,
    B_stride_width,
    O_stride_batch,
    O_stride_height,
    O_stride_width,
    batch_size,
    seq_len,
    dim,
    dim_out,
    bsx: tl.constexpr,
    bsy: tl.constexpr,
    bsk: tl.constexpr,
    group_sz: tl.constexpr,
    apply_scaling: tl.constexpr,
    scale_factor: tl.constexpr
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

    num_row_programs = tl.num_programs(1)
    num_col_programs = tl.num_programs(2)

    row_idxnew, col_idxnew = tl.swizzle2d(row_idx, col_idx, num_row_programs, num_col_programs, group_sz)

    a_offset_batch = batch_idx * A_stride_batch
    b_offset_batch = batch_idx * B_stride_batch
    output = tl.zeros((bsy, bsx), dtype=tl.float32)

    for offset in range(0, dim, bsk):
        offset_k = offset + tl.arange(0, bsk)

        # Read offsets from A_ptr
        offset_a = row_idxnew * bsy + tl.arange(0, bsy)
        mask_a = (offset_a[:, None] < seq_len) & (offset_k[None, :] < dim)
        offset_a = offset_a[:, None]*A_stride_height + offset_k[None, :]*A_stride_width  # by * bk
        a = tl.load(A_ptr + a_offset_batch + offset_a, mask_a)

        # Read offset from B_ptr
        offset_b = col_idxnew * bsx + tl.arange(0, bsx)
        mask_b = (offset_k[:, None] < dim) & (offset_b[None, :] < dim_out)
        offset_b = offset_k[:, None]*B_stride_height + offset_b[None, :]*B_stride_width  # bk * bx
        b = tl.load(B_ptr + b_offset_batch + offset_b, mask_b)

        output = tl.dot(a, b, output, allow_tf32=True) # by, bx

    offset_out_batch = batch_idx * O_stride_batch
    offset_or = row_idxnew * bsy + tl.arange(0, bsy)
    offset_oc = col_idxnew * bsx + tl.arange(0, bsx)
    offset_o = offset_or[:, None]*O_stride_height + offset_oc[None, :]*O_stride_width  # by * bx
    mask_o = (offset_or[:, None] < seq_len) & (offset_oc[None, :] < dim_out)

    if apply_scaling:
        output = scale_factor*output

    tl.store(O_ptr + offset_out_batch + offset_o, output, mask_o)


def matmul_triton(A: torch.Tensor, B: torch.Tensor, apply_scaling: bool = False, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Implements matrix multiplication between input matrix A and B
    
    Args:
        - A {torch.Tensor}: Input matrix with shape (batch_size, seq_len, dim) where B is the batch size, T is the sequence length, Cin is the input dimension
        - B {torch.Tensor}: Weight matrix with shape (batch_size, dim, dim_out) where Cout is the hidden dimension
        - apply_scaling {int}: If a scale factor should be applied to the output. 1/sqrt(dim_out) is multiplied to every element in the output

    Returns:
        - {torch.Tensor}: Output tensor with (batch_size, seq_len, dim_out)
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

    O = torch.empty((batch_size, seq_len, dim_out), device=A.device, dtype=A.dtype)

    matmul_kernel[grid](
        A, B, O,
        A_stride_batch=A.stride(0),
        A_stride_height=A.stride(1),
        A_stride_width=A.stride(2),
        B_stride_batch=B.stride(0),
        B_stride_height=B.stride(1),
        B_stride_width=B.stride(2),
        O_stride_batch=O.stride(0),
        O_stride_height=O.stride(1),
        O_stride_width=O.stride(2),
        batch_size=batch_size,
        seq_len=seq_len,
        dim=dim,
        dim_out=dim_out,
        apply_scaling=apply_scaling,
        scale_factor=scale_factor
    )

    return O


if __name__ == '__main__':
    '''
    python matmul3.py -batch_size 2 -seq_len 32 -din 16 -dout 32
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-seq_len', type=int, default=120)
    parser.add_argument('-din', type=int, default=760)
    parser.add_argument('-dout', type=int, default=500)

    args = parser.parse_args()
    print(f'Args: {args}')

    batch_size = args.batch_size
    seq_len = args.seq_len
    din = args.din
    dout = args.dout

    a = torch.randn((batch_size, seq_len, din), device=device, dtype=dtype)
    b = torch.randn((batch_size, dout, din), device=device, dtype=dtype)

    b = b.transpose(1, 2).contiguous()

    print(f'Matrix sizes: {a.shape}, {b.shape}')

    y_pytorch = torch.matmul(a, b)/math.sqrt(din)
    y_triton = matmul_triton(a, b, apply_scaling=True, scale_factor=1/math.sqrt(din))

    print(f'y_pytorch shape: {y_pytorch.shape}, y_triton shape: {y_triton.shape}')

    print(f'Original matrix:\n{a}\n{b}')
    print(f'PyTorch:\n{y_pytorch}')
    print(f'Triton:\n{y_triton}')

    # Unit testing
    assert torch.allclose(y_triton, y_pytorch, atol=1e-2, rtol=0), f"Data does not match, diff: {torch.max(torch.abs(y_pytorch-y_triton))}"

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len", "din", "dout"],
            x_vals=[64*i for i in range(1,75)],
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
    def benchmark(batch_size, seq_len, din, dout, provider):
        quantiles = [0.5, 0.2, 0.8]

        x = torch.randn((batch_size, seq_len, din), device=device, dtype=dtype)
        y = torch.randn((batch_size, dout, din), device=device, dtype=dtype)

        y = y.transpose(1, 2).contiguous()

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_triton(x, y), warmup=50, quantiles=quantiles)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(x, y), warmup=50, quantiles=quantiles)

        def gbps(ms): return 2 * seq_len * din * batch_size * 1e-12 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(
        #show_plots=True,
        print_data=True,
        save_path='./benchmarks/matmul3/'
    )
