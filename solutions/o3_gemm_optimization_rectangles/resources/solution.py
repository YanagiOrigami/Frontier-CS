import textwrap, inspect, sys, types

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def gelu(x):
            return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=2),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=8, num_stages=3),
                triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=3),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
            ],
            key=['M', 'N', 'K'],
        )
        @triton.jit
        def _matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak)
                b_ptrs = b_ptr + ((k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0)
                b = tl.load(b_ptrs, mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)

                acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

            acc = gelu(acc)

            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

        def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """
            Matrix multiplication with GELU activation using Triton.
            """
            assert a.is_cuda and b.is_cuda, "Input tensors must be on CUDA devices."
            assert a.dtype == torch.float16 and b.dtype == torch.float16, "Only float16 tensors are supported."
            M, K = a.shape
            Kb, N = b.shape
            assert K == Kb, "Incompatible dimensions."

            c = torch.empty((M, N), device=a.device, dtype=torch.float16)

            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_M']),
                triton.cdiv(N, META['BLOCK_N']),
            )

            _matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
            )
            return c
        """)
        return {"code": code}
