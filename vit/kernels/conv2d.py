import torch
import triton
import triton.language as tl

dtype = torch.float32
device = 'cuda:0'

@triton.jit
def conv2d_kernel(
    input_ptr,
    input_batch_stride,
    input_channel_stride,
    input_row_stride,
    height,
    width,
    channels,
    kernel_ptr,
    kernel_height,
    kernel_width,
    kernel_dim_stride,
    kernel_channel_stride,
    output_ptr,
    output_channel_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    for each out dim - 1st grid dim
        for each elem output - 2nd grid dim
            take 16x16x3 kernel
            take input image with the same size
            for every channel
                do dot products
    """
    batch_idx = tl.program_id(0)
    kernel_idx = tl.program_id(1)
    row_idx = tl.program_id(2)

    batch_offset = batch_idx*input_batch_stride
    # Start of output for the output data
    row_output_offset = row_idx*output_row_stride

    # Start of read from the input data
    row_offset = row_idx*kernel_height*input_row_stride

    # The nth kernel
    kernel_offset = kernel_idx*kernel_dim_stride
    kernel_offset = kernel_offset + tl.arange(0, BLOCK_SIZE)
    kernel_offset = kernel_offset[:, None] + kernel_offset[None, :]

    # Index holding the pointer to the output element in the current row
    output_col = 0
    # Iterate over the input row in small batches
    for offset in range(0, width, BLOCK_SIZE):
        elem = 0.0
        kernel_data = tl.load(kernel_ptr + kernel_offset)

        # Iterate over the channels
        for c in range(channels):
            local_row_offset = row_offset + tl.arange(0, BLOCK_SIZE)
            local_col_offset = row_offset + offset + tl.arange(0, BLOCK_SIZE)
            data_offset = local_row_offset[:, None] + local_col_offset[None, :]

            local_row_mask = local_row_offset < height
            local_col_mask = local_col_offset < width
            data_mask = local_row_mask[:, None] & local_col_mask[None, :]

            # Load input data for the current channel
            channel_offset = c*input_channel_stride
            input_data = tl.load(input_ptr + batch_offset + channel_offset + data_offset)

            # Load kernel weights for the current channel
            kernel_channel_offset = c*kernel_channel_stride
            kernel_data = tl.load(kernel_ptr + kernel_channel_offset + kernel_offset)

            dot_prdct = tl.dot(input_data, kernel_data)
            elem += tl.sum(dot_prdct)

        tl.store(output_ptr + batch_offset + row_output_offset + output_col, elem)
        output_col += 1


def conv2d_triton(
    input: torch.Tensor,
    kernel: torch.Tensor
) -> torch.Tensor:
    assert input.is_cuda and kernel.is_cuda, 'Input or kernel is not on GPU'
    assert len(input.shape) == 4, f'Input needs to be 4 dimensional, provided: {input.shape}'
    assert len(kernel.shape) == 4, f'Kernel size needs to be 3 dimensional, provided: {kernel.shape}'

    batch_size, channels, height, width = input.shape
    num_kernels, kernel_depth, kernel_height, kernel_width = kernel.shape

    assert height%kernel_height == 0 and width%kernel_width == 0, f"Input height and width should be divisible by the kernel height and width"

    output = torch.empty((batch_size, num_kernels, height//kernel_height, width//kernel_width))

    BLOCK_SIZE = 32
    # Each kernel processes a single row of the output
    grid = (batch_size, num_kernels, height//kernel_height)

    conv2d_kernel[grid](
        input_ptr=input,
        input_batch_stride=input.stride(0),
        input_channel_stride=input.stride(1),
        input_row_stride=input.stride(2),
        height=height,
        width=width,
        channels=channels,
        kernel_ptr=kernel,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        kernel_dim_stride=kernel.stride(0),
        kernel_channel_stride=kernel.stride(1),
        output_ptr=output,
        output_channel_stride=output.stride(1),
        output_row_stride=output.stride(2),
        BLOCK_SIZE=BLOCK_SIZE
    )

if __name__ == '__main__':

    batch_size=1
    height=16
    width=16
    channels=3

    kernels=5
    kernel_height=8
    kernel_width=8


    input = torch.randint(0, 5, (batch_size, channels, height, width), dtype=dtype, device=device)
    kernel = torch.randint(0, 5, (kernels, channels, kernel_height, kernel_width), dtype=dtype, device=device)

    conv_layer = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=kernels,
        kernel_size=(kernel_height, kernel_width),
        stride=(kernel_height, kernel_width),
        bias=False,
        dtype=dtype
    ).to(device)

    # For a fair comparison, copying same kernel to torch layer as well
    with torch.no_grad():
        conv_layer.weight.copy_(kernel)

    y_torch = conv_layer(input)
    y_triton = conv2d_triton(input, kernel)

    print(f'Original matrix:\n{input}')
    print(f'PyTorch patching:\n{y_torch}')
    print(f'Triton patching:\n{y_triton}')

    if torch.allclose(y_torch, y_triton):
        print('Data matches')

    else:
        print('Data does not match')

    """
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
                lambda: conv2d_kernel(x, y), quantiles=quantiles)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.add(x, y), quantiles=quantiles)

        def gbps(ms): return 2 * (x.nelement() + y.nelement()) * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)


    benchmark.run(
        show_plots=True,
        print_data=True
    )
    """