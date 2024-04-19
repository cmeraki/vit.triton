import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def patching_kernel(
    image_ptr,
    out_ptr,
    N,
    H,
    W,
    P,
    block_x: tl.constexpr,
    block_y: tl.constexpr
):
    # The current input row we are handling of the data
    row_idx = tl.program_id(axis=0)
    col_idx = tl.program_id(axis=1)

    row_offset = row_idx*block_y + tl.arange(0, block_y)
    col_offset = col_idx*block_x + tl.arange(0, block_x)
    data_offset = row_offset[None, :]*W + col_offset[:, None]

    row_mask = row_offset < H
    col_mask = col_offset < W
    data_mask = row_mask[:, None] & col_mask[None, :]

    # Read the correct path of the image
    img_r = tl.load(image_ptr+data_offset, mask=data_mask)
    img_r = tl.ravel(img_r)

    # Write to the correct nth row
    P_single_row = P*P
    num_patches_x = (W + P - 1) // P
    P_offset = (row_idx * num_patches_x + col_idx) * P_single_row

    out_offset = P_offset + tl.arange(0, block_x*block_y)
    out_mask = out_offset < N*P*P
    tl.store(out_ptr + out_offset, img_r, mask=out_mask)


def patching_triton(matrix, patch_size):
    H, W = matrix.shape
    N = int((H*W)/(patch_size*patch_size))

    # Every block will handle a 2D patch
    block_x, block_y = triton.next_power_of_2(patch_size), triton.next_power_of_2(patch_size)
    grid = lambda meta: (triton.cdiv(H, block_y), triton.cdiv(W, block_x))

    print(f'N:{N}, Block size: {(block_x, block_y)}, grid: {grid}')

    output = torch.empty(size=(N, patch_size*patch_size)).to(device=device, dtype=A.dtype)
    patching_kernel[grid](matrix, output, N=N, H=H, W=W, P=patch_size, block_x=block_x, block_y=block_y)

    return output


def patching_torch(matrix, patch_size):
    patches = matrix.unfold(
        0, patch_size, patch_size
    ).unfold(1, patch_size, patch_size)

    return patches.contiguous().view(-1, patch_size * patch_size)


if __name__ == '__main__':
    height = 16
    width = 16
    patch_size = 4
    print(f'Height: {height}, width: {width}, P: {patch_size}')

    A = torch.arange(1, height * width + 1, dtype=torch.float32).view(height, width).to(device)
    patches_pytorch = patching_torch(A, patch_size)
    patches_triton = patching_triton(A, patch_size)

    print(f'Original matrix:\n{A}')
    print(f'PyTorch patching:\n{patches_pytorch}')
    print(f'Triton patching:\n{patches_triton}')

    assert torch.allclose(patches_pytorch, patches_triton), 'Data does not match'
