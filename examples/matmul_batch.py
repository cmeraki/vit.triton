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
    BATCH_SIZE_ROW: tl.constexpr,
    BATCH_SIZE_COL: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_idx = tl.program_id(2)
