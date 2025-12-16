import torch, inspect, textwrap, tempfile, os, importlib.util, sys, triton, triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,       # *float32 / *float16 [M, N]
    targets_ptr,      # *int64 [M]
    loss_ptr,         # *float32 [M]
    M: tl.constexpr,  # int
    N: tl.constexpr,  # int
    stride_m,         # int
    stride_n,         # int
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    stride_m = tl.static_cast(stride_m, tl.int32)
    stride_n = tl.static_cast(stride_n, tl.int32)
    row_base = pid * stride_m

    offs_n = tl.arange(0, BLOCK_N)
    # ----------------------------------------------------------------------
    # 1. compute row max for numerical stability
    row_max = tl.full((), -float("inf"), dtype=tl.float32)
    col_start = 0
    while col_start < N:
        idx = col_start + offs_n
        mask = idx < N
        ptrs = logits_ptr + row_base + idx * stride_n
        x = tl.load(ptrs, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        cur_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, cur_max)
        col_start += BLOCK_N

    # ----------------------------------------------------------------------
    # 2. compute row sum(exp(logits - max))
    row_sum = tl.full((), 0.0, dtype=tl.float32)
    col_start = 0
    while col_start < N:
        idx = col_start + offs_n
        mask = idx < N
        ptrs = logits_ptr + row_base + idx * stride_n
        x = tl.load(ptrs, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        exp_x = tl.exp(x - row_max)
        row_sum += tl.sum(exp_x, axis=0)
        col_start += BLOCK_N

    logsumexp = tl.log(row_sum) + row_max

    # ----------------------------------------------------------------------
    # 3. gather target logits
    tgt_idx = tl.load(targets_ptr + pid)
    tgt_idx = tgt_idx.to(tl.int32)
    target_logit = tl.load(logits_ptr + row_base + tgt_idx * stride_n).to(tl.float32)

    # ----------------------------------------------------------------------
    # 4. compute and store loss
    loss_val = logsumexp - target_logit
    tl.store(loss_ptr + pid, loss_val)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross entropy loss using Triton kernel.
    """
    assert logits.is_cuda and targets.is_cuda, "Inputs must be CUDA tensors"
    assert logits.ndim == 2 and targets.ndim == 1, "Shapes must be (M,N) and (M,)"
    M, N = logits.shape
    out = torch.empty((M,), device=logits.device, dtype=torch.float32)
    stride_m, stride_n = logits.stride()
    grid = (M,)
    _cross_entropy_kernel[grid](
        logits,
        targets,
        out,
        M, N,
        stride_m, stride_n,
    )
    return out
'''
        return {"code": textwrap.dedent(code)}
