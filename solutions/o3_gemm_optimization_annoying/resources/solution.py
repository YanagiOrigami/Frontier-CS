import triton
import triton.language as tl
import torch


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = first_pid_m + (pid_m % GROUP_SIZE_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        b_ptrs = b_ptr + (k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(
            a_ptrs,
            mask=((offs_m[:, None] < M) & (k_offsets[None, :] < K)),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=((k_offsets[:, None] < K) & (offs_n[None, :] < N)),
            other=0.0,
        )

        acc += tl.dot(a, b)

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, "Input tensors must be 2-D"
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA device"

    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        import inspect, textwrap, types

        code_str = textwrap.dedent(
            inspect.getsource(gelu)
            + "\n"
            + inspect.getsource(_matmul_kernel)
            + "\n"
            + inspect.getsource(matmul)
        )
        # Add imports at the top
        header = "import triton\nimport triton.language as tl\nimport torch\n\n"
        full_code = header + code_str
        return {"code": full_code}
