import torch
import triton
import triton.language as tl

device = 'cuda:0'


@triton.autotune(
    configs=[
        triton.Config({'bs_row': 128, 'bs_col': 128}, num_warps=8),
        triton.Config({'bs_row': 64, 'bs_col': 64}, num_warps=8),
        triton.Config({'bs_row': 32, 'bs_col': 32}, num_warps=8),
        triton.Config({'bs_row': 16, 'bs_col': 16}, num_warps=8),
        triton.Config({'bs_row': 128, 'bs_col': 128}, num_warps=4),
        triton.Config({'bs_row': 64, 'bs_col': 64}, num_warps=4),
        triton.Config({'bs_row': 32, 'bs_col': 32}, num_warps=4),
        triton.Config({'bs_row': 16, 'bs_col': 16}, num_warps=4),
    ],
    key=['num_rows', 'num_cols']
)
@triton.jit
def add_and_norm_kernel(
    # Tensor params
    input1_ptr,
    input2_ptr,
    input_batch_stride,
    input_row_stride,
    input_col_stride,
    num_batches,
    num_rows,
    num_cols,
    out_ptr,
    # Kernel params
    bs_row: tl.constexpr,
    bs_col: tl.constexpr
):
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)

    batch_offset = batch_idx*input_batch_stride
    row_offset = row_idx*bs_row + tl.arange(0, bs_row)
    col_offset = col_idx*bs_col + tl.arange(0, bs_col)
    data_offset = row_offset[:, None] * input_row_stride + col_offset[None, :] * input_col_stride

    row_mask = row_offset < num_rows
    col_mask = col_offset < num_cols
    data_mask = row_mask[:, None] & col_mask[None, :]

    input1 = tl.load(input1_ptr + batch_offset + data_offset, data_mask)
    input2 = tl.load(input2_ptr + batch_offset + data_offset, data_mask)

    add = input1 + input2

    # Layernorm starts

    tl.store(out_ptr + batch_offset + data_offset, add, mask=data_mask)


def add_triton(
      input1: torch.Tensor,
      input2: torch.Tensor,
) -> torch.Tensor:
    """
    Performs element wise addition b/w input1 and input2

    Args:
        input1 (torch.Tensor): B * N * D
        input2 (torch.Tensor): B * N * D

    Returns:
        torch.Tensor: B * N * D
    """

    assert input1.is_contiguous() and input2.is_contiguous(), f"Input matrix needs to be contiguous"
    assert len(input1.shape) == 3, f"Only 3 dimensional input shapes are supported, provided: {input1.shape}"
    assert input1.shape == input2.shape, f"Input shapes need to be same, provided {input1.shape}, {input2.shape}"

    B, N, D = input1.shape

    out = torch.empty_like(input1, device=input1.device, dtype=input1.dtype)

    grid = lambda meta: (B, triton.cdiv(N, meta['bs_row']), triton.cdiv(D, meta['bs_col']))

    add_and_norm_kernel[grid](
        input1_ptr=input1,
        input2_ptr=input2,
        input_batch_stride=input1.stride(0),
        input_row_stride=input1.stride(1),
        input_col_stride=input1.stride(2),
        num_batches=B,
        num_rows=N,
        num_cols=D,
        out_ptr=out,
    )

    return out


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-B', type=int)
    parser.add_argument('-N', type=int)
    parser.add_argument('-D', type=int)

    args = parser.parse_args()

    batch_size=args.B
    num_tokens=args.N
    dim=args.D

    A = torch.randn(batch_size, num_tokens, dim, dtype=torch.float16, device='cuda:0')
    B = torch.randn(batch_size, num_tokens, dim, dtype=torch.float16, device='cuda:0')

    y_torch = torch.add(A, B)
    y_triton = add_triton(A, B)

    print(f'Original matrix:\n{A}\n{B}')
    print(f'PyTorch patching:\n{y_torch}')
    print(f'Triton patching:\n{y_triton}')

    if torch.allclose(y_torch, y_triton):
        print('Data matches')

    else:
        print('Data does not match')


    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],  # argument names to use as an x-axis for the plot
            # different possible values for `x_name`
            x_vals=[128*i for i in range(2, 15)],
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
            args={'B': 4, 'D': 768},  # values for function arguments not in `x_names` and `y_name`
        ))
    def benchmark(B, N, D, provider):
        x = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
        y = torch.randn(B, N, D, device='cuda', dtype=torch.float32)

        quantiles = [0.5, 0.2, 0.8]

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: add_triton(x, y), quantiles=quantiles)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.add(x, y), quantiles=quantiles)

        def gbps(ms): return 2 * x.nelement() * \
            x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)


    benchmark.run(
        show_plots=True,
        print_data=True
    )
