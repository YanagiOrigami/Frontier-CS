import math, textwrap, os, tempfile, inspect, types, importlib.util, sys, uuid, torch, triton, triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        kernel_code = r'''
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=8, num_stages=5),
    ],
    key=['M', 'N', 'K', 'dtype']
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    dtype_out: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        A = tl.load(a_ptrs, mask=mask_a, other=0.0)
        B = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # ensure dot works for all dtypes by casting to f16 if necessary
        if tl.constexpr(A.dtype is not tl.float16):
            A = A.to(tl.float16)
            B = B.to(tl.float16)

        acc += tl.dot(A, B)

    acc = gelu(acc)
    C = acc.to(dtype_out)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, C, mask=mask_c)

def _to_tl_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return tl.float16
    if torch_dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.
    Args:
        a: (M, K) tensor
        b: (K, N) tensor
    Returns:
        (M, N) tensor with GELU applied
    """
    assert a.ndim == 2 and b.ndim == 2, "Input tensors must be 2-D"
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes"
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA device"

    M, K = a.shape
    K2, N = b.shape
    assert K2 == K

    dtype_out = a.dtype  # assume both inputs have same dtype
    C = torch.empty((M, N), device=a.device, dtype=dtype_out)

    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))  # will be tuned by autotuner anyway

    _matmul_kernel[grid](
        a, b, C,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        C.stride(0), C.stride(1),
        _to_tl_dtype(dtype_out)
    )
    return C
'''
        return {"code": textwrap.dedent(kernel_code)}
