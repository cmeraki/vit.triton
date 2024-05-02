import torch
import triton
import triton.language as tl

device = 'cuda'
dtype = torch.float32

@triton.jit
def layernorm_kernel(
    a_ptr,
    a_stride_m,
    a_stride_n,
    N, # Number of columns
    weight_ptr,
    bias_ptr,
    eps,
    out_ptr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    row = row_idx * a_stride_m
    #mask = tl.arange(0, BLOCK_SIZE) < N 

    local_sum = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        local_offset = row + offset + tl.arange(0, BLOCK_SIZE)    
        mask = offset + tl.arange(0, BLOCK_SIZE) < N
        data = tl.load(a_ptr + local_offset, mask=mask, other=0.0)

        local_sum += tl.sum(data)

    mean = local_sum/N

    local_std = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        local_offset = row + offset + tl.arange(0, BLOCK_SIZE)
        mask = offset + tl.arange(0, BLOCK_SIZE) < N
        data = tl.load(a_ptr + local_offset, mask=mask, other=mean)

        x = data-mean
        x = x*x 

        local_std += tl.sum(x)

    std = local_std / N + eps
    std = tl.sqrt(std)

    for offset in range(0, N, BLOCK_SIZE):
        local_offset = offset + tl.arange(0, BLOCK_SIZE)
        mask = local_offset < N
        w = tl.load(weight_ptr + local_offset, mask=mask, other=0.0)
        b = tl.load(bias_ptr + local_offset, mask=mask, other=0.0)

        local_offset += row
        mask = offset + tl.arange(0, BLOCK_SIZE) < N
        x = tl.load(a_ptr + local_offset, mask=mask, other=0.0)

        norm = w*((x-mean)/std) + b

        tl.store(out_ptr+local_offset, norm, mask=mask)

    #data = tl.load(a_ptr + row + tl.arange(0, BLOCK_SIZE), tl.arange(0, BLOCK_SIZE) < N)
    #print('rowidx, row, data, mean, std\t', row_idx, row, data, mean, std)

def layernorm_triton(A: torch.Tensor, weight, bias, eps):
    assert A.is_contiguous(), 'Matrix is not contiguous'
    assert A.is_cuda, 'Matrix is not on GPU'

    # Output tensor
    O = torch.empty_like(A, device='cuda', dtype=torch.float32)
    #BLOCK_SIZE_X = triton.next_power_of_2(A.shape[1]) # every block will process a complete row
    BLOCK_SIZE_X = 16
    grid = (M, )

    print(f'Grid: {grid}, block: {BLOCK_SIZE_X}')

    layernorm_kernel[grid](
        A, A.stride(0), A.stride(1), A.shape[1], weight, bias, eps,
        O, BLOCK_SIZE=BLOCK_SIZE_X
    )

    return O


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-M', type=int)
    parser.add_argument('-N', type=int)

    args = parser.parse_args()

    M = args.M
    N = args.N

    a = torch.randint(0, 10, (M, N), device=device, dtype=dtype)
    _shape = (a.shape[-1], )
    weight = torch.randn(_shape, device=device, dtype=dtype)
    bias = torch.randn(_shape, device=device, dtype=dtype)
    eps = 1e-5

    y_pytorch = torch.nn.functional.layer_norm(a, _shape, weight, bias, eps).to(dtype)
    y_triton = layernorm_triton(a, weight=weight, bias=bias, eps=eps)

    print(f'Original tensor\n{a}')
    print(f'PyTorch layer norm\n{y_pytorch}')
    print(f'Triton layer norm\n{y_triton}')
    
    assert torch.allclose(y_triton, y_pytorch, atol=1e-2, rtol=0), 'Data does not match'
    
