import triton
import triton.language as tl
import torch
import math
from typing import Dict


@triton.jit
def _scan_chunk_kernel(
    X_ptr,
    A_ptr,
    B_ptr,
    Y_ptr,
    prev_ptr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < D

    y_prev = tl.load(prev_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)

    for t in range(CHUNK):
        ptr = t * D + offs_d
        x_t = tl.load(X_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
        a_t = tl.load(A_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
        b_t = tl.load(B_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
        y_t = a_t * y_prev + b_t * x_t
        tl.store(Y_ptr + ptr, y_t.to(tl.float16), mask=mask)
        y_prev = y_t


def chunk_scan(
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    chunk: int = 128,
    BD: int = 128,
) -> torch.Tensor:
    assert X.shape == A.shape == B.shape, "Shapes of X, A, B must match."
    assert X.dtype == torch.float16 and A.dtype == torch.float16 and B.dtype == torch.float16
    assert X.is_cuda and A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
    L, D = X.shape
    assert L % chunk == 0, "Sequence length L must be divisible by chunk size"

    Y = torch.empty_like(X)
    zeros_row = torch.zeros(D, dtype=torch.float16, device=X.device)

    num_chunks = L // chunk
    grid_d = lambda BLOCK_D: (triton.cdiv(D, BLOCK_D),)

    for idx in range(num_chunks):
        start = idx * chunk
        X_chunk = X[start : start + chunk]
        A_chunk = A[start : start + chunk]
        B_chunk = B[start : start + chunk]
        Y_chunk = Y[start : start + chunk]

        prev_ptr = zeros_row if idx == 0 else Y[start - 1]

        _scan_chunk_kernel[grid_d(BD)](
            X_chunk,
            A_chunk,
            B_chunk,
            Y_chunk,
            prev_ptr,
            D=D,
            CHUNK=chunk,
            BLOCK_D=BD,
            num_warps=4,
        )

    return Y


class Solution:
    def solve(self, spec_path: str = None) -> Dict[str, str]:
        import os
        return {"program_path": os.path.abspath(__file__)}
