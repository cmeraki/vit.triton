import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def patching_kernel(
    image_ptr,
    out_ptr,
    N,
    W,
    P,
    BLOCK_SIZE: tl.constexpr,
):
    # The current input row we are handling of the data
    row_idx = tl.program_id(axis=0)
    col_idx = tl.program_id(axis=1)
    row_offset = row_idx * BLOCK_SIZE * W//P
    col_offset = col_idx * BLOCK_SIZE

    # Read the correct path of the image
    read_offset = row_offset + col_offset + tl.arange(0, BLOCK_SIZE)
    mask_offset = read_offset < W*W

    img_r = tl.load(image_ptr+read_offset, mask=mask_offset)

    # Write to the correct path of the image
    P_row = (row_idx)%P
    P_col = col_idx
    P_single_row = P*P

    P_offset = P_single_row*P_col + P_single_row*(W//P)*row_idx
    P_start = P_offset + P*P_row

    out_offset = P_start + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offset < N*P*P
    tl.store(out_ptr + out_offset, img_r, mask=out_mask)


def patching_triton(matrix, patch_size):
    H, W = matrix.shape
    # N*P.P
    N = int((H*W)/(patch_size*patch_size))
    # Every block will handle single row of the resulting patch
    BLOCK_SIZE = triton.next_power_of_2(patch_size)
    # Grid is parallelized across num rows and num patches horizontally
    grid = (H,W//N)
    print(f'N:{N}, Block size: {BLOCK_SIZE}, grid: {grid}')

    output = torch.empty(size=(N, patch_size*patch_size)).to(device=device, dtype=A.dtype)
    patching_kernel[grid](matrix, output, N=N, W=W, P=patch_size, BLOCK_SIZE=BLOCK_SIZE)

    return output


def patching_torch(matrix, patch_size):
    patches = matrix.unfold(
        0, patch_size, patch_size
    ).unfold(1, patch_size, patch_size)

    return patches.contiguous().view(-1, patch_size * patch_size)


if __name__ == '__main__':
    height = 8
    width = 8
    patch_size = 4
    print(f'Height: {height}, width: {width}, P: {patch_size}')

    A = torch.arange(1, height * width + 1, dtype=torch.float32).view(height, width).to(device)
    patches_pytorch = patching_torch(A, patch_size)
    patches_triton = patching_triton(A, patch_size)

    print(f'Original matrix:\n{A}')
    print(f'PyTorch patching:\n{patches_pytorch}')
    print(f'Triton patching:\n{patches_triton}')
