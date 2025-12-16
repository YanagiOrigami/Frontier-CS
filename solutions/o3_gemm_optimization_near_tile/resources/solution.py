import textwrap, json, inspect, types, sys, os, re, math, functools, random, itertools, collections, typing, dataclasses, pathlib, builtins
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def gelu(x):
            return x * 0.5 * (1.0 + tl.libdevice.erf(x * 0.7071067811865476))

        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
            ],
            key=['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn']
        )
        @triton.jit
        def _matmul_kernel(
            A_ptr, B_ptr, C_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            *,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr,
            OUTPUT_DTYPE: tl.constexpr
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            offs_am = offs_m[:, None] * stride_am
            offs_bn = offs_n[None, :] * stride_bn

            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)

                a_ptrs = A_ptr + offs_am + offs_k[None, :] * stride_ak
                b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn

                a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

                a = a.to(tl.float32)
                b = b.to(tl.float32)

                acc += tl.dot(a, b)

            acc = gelu(acc)

            if tl.constexpr(OUTPUT_DTYPE == tl.float16):
                acc = acc.to(tl.float16)
            elif tl.constexpr(OUTPUT_DTYPE == tl.bfloat16):
                acc = acc.to(tl.bfloat16)

            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

        def _get_output_dtype(dtype: torch.dtype):
            if dtype == torch.float16:
                return tl.float16
            if dtype == torch.bfloat16:
                return tl.bfloat16
            return tl.float32

        def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            assert a.ndim == 2 and b.ndim == 2, 'Only 2D tensors supported'
            M, K = a.shape
            K2, N = b.shape
            assert K == K2, 'Incompatible dimensions'
            assert a.is_cuda and b.is_cuda, 'Tensors must be on CUDA device'
            dtype = a.dtype
            c = torch.empty((M, N), device=a.device, dtype=dtype)

            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_M']),
                triton.cdiv(N, META['BLOCK_N'])
            )

            _matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                OUTPUT_DTYPE=_get_output_dtype(dtype)
            )
            return c
        """)
        return {"code": code}
