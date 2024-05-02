import torch
import triton
import triton.language as tl

device = 'cuda:0'

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
    # Layernorm params
    weight_ptr,
    bias_ptr,
    eps,
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
      weight: torch.Tensor=None,
      bias: torch.Tensor=None,
      eps: float=None
) -> torch.Tensor:
    """
    Performs element wise addition b/w input1 and input2
    and performs layer norm on the output

    Args:
        input1 (torch.Tensor): _description_
        input2 (torch.Tensor): _description_
        weight (torch.Tensor): _description_
        bias (torch.Tensor): _description_
        eps (float): _description_

    Returns:
        torch.Tensor: _description_
    """

    assert input1.is_contiguous() and input2.is_contiguous(), f"Input matrix needs to be contiguous"
    assert len(input1.shape) == 3, f"Only 3 dimensional input shapes are supported, provided: {input1.shape}"
    assert input1.shape == input2.shape, f"Input shapes need to be same, provided {input1.shape}, {input2.shape}"

    B, N, D = input1.shape

    out = torch.empty_like(input1, device=input1.device, dtype=input1.dtype)
    bs_row, bs_col = 16, 16 

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
        weight_ptr=None,
        bias_ptr=None,
        eps=None,
        bs_row=bs_row,
        bs_col=bs_col
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


