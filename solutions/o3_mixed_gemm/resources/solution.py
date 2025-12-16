import textwrap

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {
            "code": textwrap.dedent(
                """
                import torch
                import triton
                import triton.language as tl

                @triton.jit
                def _linear_gelu_kernel(
                    X_ptr, W_ptr, B_ptr, O_ptr,
                    M, N, K,
                    stride_xm, stride_xk,
                    stride_wk, stride_wn,
                    stride_om, stride_on,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr,
                ):
                    pid_m = tl.program_id(axis=0)
                    pid_n = tl.program_id(axis=1)

                    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                    offs_k = tl.arange(0, BLOCK_K)

                    X_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
                    W_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

                    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                    for k0 in range(0, K, BLOCK_K):
                        x = tl.load(
                            X_ptrs,
                            mask=(offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K),
                            other=0.0,
                        )
                        w = tl.load(
                            W_ptrs,
                            mask=((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N),
                            other=0.0,
                        )
                        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))
                        X_ptrs += BLOCK_K * stride_xk
                        W_ptrs += BLOCK_K * stride_wk

                    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
                    acc += bias[None, :]

                    acc = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))

                    O_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
                    tl.store(
                        O_ptrs,
                        acc.to(tl.float16),
                        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                    )

                def linear_gelu(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                    assert X.is_cuda and W.is_cuda and B.is_cuda, "All tensors must be on CUDA"
                    assert X.dtype == torch.float16 and W.dtype == torch.float16, "X and W must be float16"
                    assert B.dtype == torch.float32, "B must be float32"
                    M, K = X.shape
                    Kw, N = W.shape
                    assert K == Kw, "Incompatible dimensions between X and W"

                    O = torch.empty((M, N), device=X.device, dtype=torch.float16)

                    BLOCK_M = 128
                    BLOCK_N = 64
                    BLOCK_K = 32

                    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
                    _linear_gelu_kernel[grid](
                        X, W, B, O,
                        M, N, K,
                        X.stride(0), X.stride(1),
                        W.stride(0), W.stride(1),
                        O.stride(0), O.stride(1),
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                        num_warps=4,
                        num_stages=3,
                    )
                    return O
                """
            )
        }
