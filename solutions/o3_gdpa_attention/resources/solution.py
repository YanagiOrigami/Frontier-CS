import math, torch, torch.nn.functional as F, triton, triton.language as tl

def gdpa_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, GQ: torch.Tensor, GK: torch.Tensor) -> torch.Tensor:
    Qg = Q * torch.sigmoid(GQ)
    Kg = K * torch.sigmoid(GK)
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(Qg, Kg, V, dropout_p=0.0, is_causal=False)
    attn = torch.matmul(Qg, Kg.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": inspect.getsource(gdpa_attn)}
