import torch, math, triton, triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        kernel_code = '''
import torch
import triton
import triton.language as tl
import math

def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """
    Flash attention computation with optional causal masking.
    Args:
        Q: (Z, H, M, Dq) - Query tensor (float16)
        K: (Z, H, N, Dq) - Key tensor (float16)
        V: (Z, H, N, Dv) - Value tensor (float16)
        causal: Apply causal masking if True
    Returns:
        (Z, H, M, Dv) - Attention output (float16)
    """
    Dq = Q.shape[-1]
    scale = 1.0 / math.sqrt(Dq)

    # Compute scaled dot-product attention scores in float32 for stability
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores = scores.float()

    if causal:
        M = Q.shape[-2]
        N = K.shape[-2]
        if N == M:
            mask = torch.triu(torch.ones((M, N), dtype=torch.bool, device=scores.device), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
        else:
            i_idx = torch.arange(M, device=scores.device)[:, None]
            j_idx = torch.arange(N, device=scores.device)[None, :]
            mask = j_idx > i_idx
            scores = scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1).to(Q.dtype)
    out = torch.matmul(attn, V)
    return out
'''
        return {"code": kernel_code}
