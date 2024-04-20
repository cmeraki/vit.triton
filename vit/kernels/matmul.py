import torch
import triton
import triton.language as tl

device = 'cuda:0'


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, O_ptr,
    A_stride_height, A_stride_width,
    B_stride_height, B_stride_width,
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
    # Load apt data into memory
    row_idx = tl.program_id(axis=0)
    col_idx = tl.program_id(axis=1)

    offset_k = tl.arange(0, K)

    # Read offsets from A_ptr
    offset_a = row_idx * bsy + tl.arange(0, bsy)
    offset_a = offset_a[:, None]*A_stride_height + offset_k[None, :]*A_stride_width  # by * K
    mask_a = row_idx * bsy + tl.arange(0, bsy)
    mask_a = (mask_a[:, None] < M) & (offset_k[None, :] < K)
    a = tl.load(A_ptr + offset_a)

    # Read offset from B_ptr
    offset_b = col_idx * bsx + tl.arange(0, bsx)
    offset_b = offset_k[:, None]*B_stride_height + offset_b[None, :]*B_stride_width  # K * bx
    mask_b = col_idx * bsx + tl.arange(0, bsx)
    mask_b = (offset_k[:, None] < K) & (mask_b[None, :] < N)
    b = tl.load(B_ptr + offset_b)

    o = tl.dot(a, b)

    offset_or = row_idx * bsy + tl.arange(0, bsy)
    offset_oc = col_idx * bsx + tl.arange(0, bsx)
    offset_o = offset_or[:, None]*O_stride_height\
        + offset_oc[None, :]*O_stride_width  # by * bx
    mask_o = (offset_or[:, None] < M) & (offset_oc[None, :] < N)

    tl.store(O_ptr+offset_o, o)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K, N = B.shape

    m, n = 16, 16
    bsx, bsy = triton.next_power_of_2(n), triton.next_power_of_2(m)
    grid = (triton.cdiv(M, bsy), triton.cdiv(N, bsx))
    print(f'bsx {bsx}, bsy {bsy}, grid {grid}')

    O = torch.empty((M, N)).to(device)
    matmul_kernel[grid](
        A, B, O,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        O.stride(0), O.stride(1),
        M, N, K, bsx=bsx, bsy=bsy
    )

    return O


if __name__ == '__main__':
    '''
    python matmul.py -M 512 -N 512 -K 512
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-M', type=int)
    parser.add_argument('-K', type=int)
    parser.add_argument('-N', type=int)

    args = parser.parse_args()
    print(f'Args: {args}')
    M = args.M
    K = args.K
    N = args.N

    A = torch.randint(0, 10, (M, K), device='cuda', dtype=torch.float32)
    B = torch.randint(0, 5, (K, N), device='cuda', dtype=torch.float32)

    assert A.shape[1] == B.shape[0], 'Matrix are not compatible for multiplication'

    y_pytorch = torch.matmul(A, B)
    y_triton = matmul_triton(A, B)

    # print(y_pytorch.shape, y_triton.shape)

    # print(f'Original matrix:\n{A}\n{B}')
    # print(f'PyTorch:\n{y_pytorch}')
    # print(f'Triton:\n{y_triton}')

    assert torch.allclose(y_pytorch, y_triton), 'Data does not match'

    print('Tensors match')
