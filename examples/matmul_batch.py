import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    input_ptr,
    input_batch_stride,
    input_row_stride,
    input_col_stride,
    weight_ptr,
    weight_row_stride,
    weight_col_stride,
    output_ptr,
    output_batch_stride,
    output_row_stride,
    output_col_stride,
    num_rows,
    num_input_cols: tl.constexpr,
    num_output_cols,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    # Getting block indexes in all 3 dimensions
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_idx = tl.program_id(2)

    # Offsets for input data
    input_batch_offset = batch_idx * input_batch_stride

    input_row_offset = row_idx*BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    input_row_mask = input_row_offset[:, None] < num_rows
    input_row_offset = input_row_offset[:, None] * input_row_stride # Selecting relevant rows from input

    input_col_offset = tl.arange(0, num_input_cols)
    input_col_mask = input_col_offset[None, :] < num_input_cols
    input_col_offset = input_col_offset[None, :] * input_col_stride # Selecting all columns from input

    input_data_ptr = input_ptr + input_batch_offset + input_row_offset + input_col_offset
    input_data = tl.load(input_data_ptr, mask=(input_row_mask & input_col_mask)) # BLOCK_SIZE_ROW x D_in

    # Offsets for weight data
    weight_row_offset = tl.arange(0, num_input_cols)
    weight_row_mask = weight_row_offset[:, None] < num_input_cols
    weight_row_offset = weight_row_offset[:, None] * weight_row_stride # Selecing all rows from weight

    weight_col_offset = col_idx*BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    weight_col_mask = weight_col_offset < num_output_cols
    weight_col_offset = weight_col_offset[None, :] * weight_col_stride # Selecting relevant columns from input

    weight_data_ptr = weight_ptr + weight_row_offset + weight_col_offset
    weight_data = tl.load(weight_data_ptr, mask=(weight_row_mask & weight_col_mask)) # D_in x BLOCK_SIZE_COL

    # Computation
    result = tl.dot(input_data, weight_data) # Matmul of a small block, BLOCK_SIZE_ROW x BLOCK_SIZE_COL

    # Offsets for output data
    output_batch_offset = batch_idx * output_batch_stride

    output_row_offset = row_idx*BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    output_row_mask = output_row_offset[:, None] < num_rows
    output_row_offset = output_row_offset[:, None] * output_row_stride

    output_col_offset = col_idx*BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    output_col_mask = output_col_offset[None, :] < num_output_cols
    output_col_offset = output_col_offset[None, :] * output_col_stride

    output_data_ptr = output_ptr + output_batch_offset + output_row_offset + output_col_offset
    tl.store(output_data_ptr, result, mask=(output_row_mask & output_col_mask))


def matmul(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

    assert input.is_cuda, 'Inputs are not on GPU, ensure the input matrix is loaded on the GPU'
    assert weight.is_cuda, 'Weights are not on GPU, ensure the weight matrix is loaded on the GPU'
    assert input.shape[-1] == weight.shape[-2], 'Input and weight matrix are not compatible'

    B, N, D_in = input.shape
    _, D_out = weight.shape

    output = torch.empty((B, N, D_out), device=input.device, dtype=input.dtype)

    BLOCK_SIZE_ROW, BLOCK_SIZE_COL = 16, 16
    # We are launching the grid to align with the output
    grid = lambda meta: (B, triton.cdiv(N, meta["BLOCK_SIZE_ROW"]), triton.cdiv(D_out, meta["BLOCK_SIZE_COL"]))

    matmul_kernel[grid](
        input_ptr=input,
        input_batch_stride=input.stride(0),
        input_row_stride=input.stride(1),
        input_col_stride=input.stride(2),
        weight_ptr=weight,
        weight_row_stride=weight.stride(0),
        weight_col_stride=weight.stride(1),
        output_ptr=output,
        output_batch_stride=output.stride(0),
        output_row_stride=output.stride(1),
        output_col_stride=output.stride(2),
        num_rows=N,
        num_input_cols=D_in,
        num_output_cols=D_out,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

    return output


if __name__ == '__main__':
    batch_size = 4
    # All the following dims need to be greater than 16
    seq_len = 40
    d_in = 32 # This needs to be a power of 2, this is only a requirement for this program. We can optimize matmul in kernel to remove this condition
    d_out = 30
    device = 'cuda:0'
    dtype = torch.float32

    a = torch.randint(0, 10, (batch_size, seq_len, d_in), device=device, dtype=dtype)
    w = torch.randint(0, 5, (d_in, d_out), device=device, dtype=dtype)

    out_triton = matmul(a, w)
    out_torch = torch.matmul(a, w)

    print(f'Original tensors: \nA\n{a}, \nW\n{w}')
    print(f'Torch:\n{out_torch}')
    print(f'Triton: \n{out_triton}')
    assert torch.allclose(out_triton, out_torch, atol=1e-2), "Torch and triton implementation does not match"
