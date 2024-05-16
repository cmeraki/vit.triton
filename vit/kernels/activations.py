"""
From: https://github.com/BobMcDear/attorch/blob/main/attorch/act_kernels.py#L99
"""

import triton
import triton.language as tl

@triton.jit
def gelu(input):
    """
    Applies GELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input
