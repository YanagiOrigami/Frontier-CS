import math
import torch
import triton
import triton.language as tl


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = r'''
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _decoding_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    H, M, scale,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_VALUE: tl.constexpr,
):
    pid = tl.program_id(0)
    h_id = pid % H
    m_id = (pid // H) % M
    z_id = pid // (H * M)

    offs_dq = tl.arange(0, D_HEAD)
    offs_dv = tl.arange(0, D_VALUE)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q
    q_ptrs = Q_ptr + z_id * stride_qz + h_id * stride_qh + m_id * stride_qm + offs_dq * stride_qd
    q = tl.load(q_ptrs).to(tl.float32)

    # Accumulators
    max_val = tl.full((), -1e9, tl.float32)
    denom = tl.zeros((), tl.float32)
    numer = tl.zeros((D_VALUE,), tl.float32)

    k_head_ptr = K_ptr + z_id * stride_kz + h_id * stride_kh
    v_head_ptr = V_ptr + z_id * stride_vz + h_id * stride_vh

    n_start = tl.zeros((), tl.int32)
    while n_start < N:
        n_idx = n_start + offs_n
        mask = n_idx < N

        k_ptrs = k_head_ptr + n_idx[:, None] * stride_kn + offs_dq[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=mask[:, None]).to(tl.float32)

        scores = tl.sum(k_block * q[None, :], axis=1) * scale
        scores = tl.where(mask, scores, -1e9)

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        prev_factor = tl.exp(max_val - new_max)
        exp_scores = tl.exp(scores - new_max)

        part_denom = tl.sum(exp_scores, axis=0)

        v_ptrs = v_head_ptr + n_idx[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=mask[:, None]).to(tl.float32)

        part_numer = tl.sum(exp_scores[:, None] * v_block, axis=0)

        numer = numer * prev_factor + part_numer
        denom = denom * prev_factor + part_denom
        max_val = new_max

        n_start += BLOCK_N

    out = numer / denom
    o_ptrs = O_ptr + z_id * stride_oz + h_id * stride_oh + m_id * stride_om + offs_dv * stride_od
    tl.store(o_ptrs, out.to(tl.float16))


def decoding_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Scaled dot-product attention for decoding (usually M=1).
    Q: (Z, H, M, Dq)
    K: (Z, H, N, Dq)
    V: (Z, H, N, Dv)
    Returns: (Z, H, M, Dv)
    """
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
    Z, H, M, Dq = Q.shape
    _, _, N, _ = K.shape
    _, _, _, Dv = V.shape

    O = torch.empty((Z, H, M, Dv), device=Q.device, dtype=torch.float16)

    BLOCK_N = 128
    scale = 1.0 / math.sqrt(Dq)

    grid = (Z * H * M,)

    _decoding_kernel[grid](
        Q, K, V, O,
        N,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        H, M, scale,
        BLOCK_N=BLOCK_N,
        D_HEAD=Dq,
        D_VALUE=Dv,
        num_warps=4,
        num_stages=2,
    )

    return O
'''
        return {"code": code}
