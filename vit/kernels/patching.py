import sys
import torch
import triton
import triton.language as tl

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="INFO")

device = 'cuda:0'

@triton.jit
def patching_kernel(
    image_ptr,
    out_ptr,
    batch_size,
    batch_stride,   # Stride to move to the next elem in batch
    image_stride0,  # Stride to move to the next channel
    image_stride1,  # Stride to move to the next row
    N,
    H,
    W,
    C,
    P,
    block_x: tl.constexpr,
    block_y: tl.constexpr
):
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)

    batch_offset = batch_idx * batch_stride
    row_offset = row_idx*block_y + tl.arange(0, block_y)
    col_offset = col_idx*block_x + tl.arange(0, block_x)
    data_offset = row_offset[None, :]*image_stride1 + col_offset[:, None] # P x P

    row_mask = row_offset < H
    col_mask = col_offset < W
    data_mask = row_mask[:, None] & col_mask[None, :]

    # Read the correct path of the image
    img_r = tl.load(image_ptr + batch_offset + data_offset, mask=data_mask)
    img_g = tl.load(image_ptr + batch_offset + data_offset + image_stride0, mask=data_mask)
    img_b = tl.load(image_ptr + batch_offset + data_offset + image_stride0*2, mask=data_mask)

    # Write to the correct nth row
    P_single_row = P*P*C
    num_patches_x = (W + P - 1) // P
    P_offset = (row_idx * num_patches_x + col_idx) * P_single_row

    out_offset = P_offset + tl.arange(0, block_x*block_y)
    out_mask = out_offset < N*P*P*C # Dimension of a single patch multiplied with the number of patches

    tl.store(out_ptr + batch_offset + out_offset, tl.ravel(img_r), mask=out_mask)
    tl.store(out_ptr + batch_offset + out_offset + P*P, tl.ravel(img_g), mask=out_mask)
    tl.store(out_ptr + batch_offset + out_offset + P*P*2, tl.ravel(img_b), mask=out_mask)


def patching_triton(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Patching function which calls triton kernel

    Args
        - image {torch.Tensor}: A 4D matrix with dimensions (B, C, H, W) where B is the batch size
        - patch_size {int}: Patch size

    Returns
        - {torch.Tensor}: A 3D matrix with dimensions (B, N, P^2.C) where N is the number of patches per image
    """
    assert len(image.shape) == 4, "The provided matrix for patching should be 4 dimensional (B, C, H, W)"

    B, C, H, W = image.shape
    N = int((H*W)/(patch_size*patch_size)) # Number of tokens per image

    # The grid and blocks are arranged based on the input image
    # Every block will handle a 2D patch
    block_x, block_y = triton.next_power_of_2(patch_size), triton.next_power_of_2(patch_size)
    grid = lambda meta: (B, triton.cdiv(H, block_y), triton.cdiv(W, block_x))

    logger.info(f'Patching kernel - N:{N}, Block size: {(block_x, block_y)}, grid: {grid}, image stride: {image.stride()}')

    output = torch.empty(size=(B, N, patch_size*patch_size*C)).to(device=image.device, dtype=image.dtype)
    patching_kernel[grid](
            image,
            output,
            batch_size=B,
            batch_stride=image.stride(0),
            image_stride0=image.stride(1),
            image_stride1=image.stride(2),
            N=N,
            H=H,
            W=W,
            C=C,
            P=patch_size,
            block_x=block_x,
            block_y=block_y
    )

    return output


def patching_torch(matrix: torch.Tensor, patch_size: int) -> torch.Tensor:
    patches = matrix.unfold(
        2, patch_size, patch_size
    ).unfold(
        3, patch_size, patch_size
    )

    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(matrix.size(0), -1, matrix.size(1) * patch_size * patch_size)
    
    return patches


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-B', type=int)
    parser.add_argument('-H', type=int)
    parser.add_argument('-W', type=int)
    parser.add_argument('-P', type=int)
    args = parser.parse_args()

    batch_size = args.B
    height = args.H
    width = args.W
    channels = 3
    patch_size = args.P

    print(f'Batch size: {batch_size}, Height: {height}, width: {width}, channels: {channels}, P: {patch_size}')

    A = torch.arange(1, batch_size * height * width * channels + 1, dtype=torch.float32).view(batch_size, channels, height, width).to(device)
    patches_pytorch = patching_torch(A, patch_size)
    patches_triton = patching_triton(A, patch_size)

    print(f'Original matrix:\n{A}')
    print(f'PyTorch patching:\n{patches_pytorch}')
    print(f'Triton patching:\n{patches_triton}')

    assert torch.allclose(patches_pytorch, patches_triton), 'Data does not match'
