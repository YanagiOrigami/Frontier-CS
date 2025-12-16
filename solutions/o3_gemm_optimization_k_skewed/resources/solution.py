import textwrap, inspect, sys, types

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl


        @triton.jit
        def gelu(x):
            # 0.7071067811865476 = 1/sqrt(2)
            return x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))


        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
                triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
                triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
                triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
                triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
            ],
            key=["M", "N", "K"],
        )
        @triton.jit
        def _matmul_kernel(
            A_ptr, B_ptr, C_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            k = 0
            while k < K:
                mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
                mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                acc += tl.dot(a.to(tl.float16), b.to(tl.float16))
                k += BLOCK_K
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

            acc = gelu(acc)
            acc = acc.to(tl.float16)

            c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


        def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            \"\"\"
            Matrix multiplication with GELU activation using a Triton kernel.
            \"\"\"
            assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D matrices"
            M, K = a.shape
            Kb, N = b.shape
            assert K == Kb, "Inner dimensions must agree"
            assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
            assert a.dtype == b.dtype, "Input tensors must share dtype"

            if a.dtype != torch.float16:
                # Fallback to native torch for unsupported dtypes
                return torch.nn.functional.gelu(a @ b)

            a_contig = a.contiguous()
            b_contig = b.contiguous()
            out = torch.empty((M, N), device=a.device, dtype=a.dtype)

            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
            )

            _matmul_kernel[grid](
                a_contig, b_contig, out,
                M, N, K,
                a_contig.stride(0), a_contig.stride(1),
                b_contig.stride(0), b_contig.stride(1),
                out.stride(0), out.stride(1),
            )
            return out
        """)
        return {"code": code}
