import torch
import triton
import triton.language as tl

dtype = torch.float32
device = 'cuda:0'


@triton.jit
def softmax_kernel(
    input_ptr,
    input_batch_stride,
    input_row_stride,
    output_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(axis=0)
    row_id = tl.program_id(axis=1)

    batch_offset = batch_id * input_batch_stride
    row_offset = row_id * input_row_stride + tl.arange(0, BLOCK_SIZE)

    mask = tl.arange(0, BLOCK_SIZE) < num_cols
    data = tl.load(input_ptr + batch_offset + row_offset, mask, other=-float('inf'))
    data = data - tl.max(data, axis=0)

    row_wise_exp = tl.exp(data)
    row_wise_sum = tl.sum(row_wise_exp, axis=0)
    output = row_wise_exp/row_wise_sum

    tl.store(output_ptr + batch_offset + row_offset, output, mask=mask)


def softmax_triton(A: torch.Tensor) -> torch.Tensor:
    """
    Performs softmax on input. This function always performs softmax on the last axis
    of the input

    Args:
        A (torch.Tensor): Input matrix of size (B * N * D)

    Returns:
        {torch.Tensor}: Ouput tensor is of the same shape (B * N * D)
    """
    assert A.is_cuda, "Input is not on GPU"
    assert len(A.shape) == 3, f"Input needs to be 3 dimensional, provided: {A.shape}"

    batch, rows, cols = A.shape

    output = torch.empty_like(A)

    BLOCK_SIZE = triton.next_power_of_2(cols)
    grid = (batch, rows, )

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_kernel[grid](
        input_ptr=A,
        input_batch_stride=A.stride(0),
        input_row_stride=A.stride(1),
        output_ptr=output,
        num_rows=rows,
        num_cols=cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return output


if __name__ == '__main__':
    A = torch.randint(0, 10, size=(1, 1823, 781), device=device, dtype=dtype)

    y_torch = torch.softmax(A, dim=-1)
    y_triton = softmax_triton(A)

    assert torch.allclose(y_triton, y_torch), "Data is not the same"

    print("Data is same")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['D'],
            x_vals=[128 * i for i in range(2, 100)],
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
            args={'B': 4, 'N': 4096},  # values for function arguments not in `x_names` and `y_name`
        ))
    def benchmark(B, N, D, provider):
        x = torch.randn(B, N, D, device=device, dtype=dtype)

        quantiles = [0.5, 0.2, 0.8]

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_triton(x), quantiles=quantiles)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1), quantiles=quantiles)

        def gbps(ms): return 2 * x.nelement() * \
            x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(
        show_plots=True,
        print_data=True
    )
