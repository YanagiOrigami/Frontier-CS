import textwrap

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl
        
        def fused_linear_jsd(X: torch.Tensor, 
                             W1: torch.Tensor, B1: torch.Tensor, 
                             W2: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
            \"\"\"
            Fused computation of two linear layers followed by Jensen-Shannon Divergence.
            All inputs are expected on the same CUDA device.
            \"\"\"
            # First linear projection (in float16), then cast to float32 for stability
            logits1 = torch.matmul(X, W1).to(torch.float32) + B1
            # Second linear projection
            logits2 = torch.matmul(X, W2).to(torch.float32) + B2

            # log-softmax for first logits
            max1 = logits1.max(dim=1, keepdim=True).values
            lse1 = torch.log(torch.exp(logits1 - max1).sum(dim=1, keepdim=True)) + max1
            logp1 = logits1 - lse1

            # log-softmax for second logits
            max2 = logits2.max(dim=1, keepdim=True).values
            lse2 = torch.log(torch.exp(logits2 - max2).sum(dim=1, keepdim=True)) + max2
            logp2 = logits2 - lse2

            # Convert to probabilities
            p1 = torch.exp(logp1)
            p2 = torch.exp(logp2)

            # Mixture distribution
            m = 0.5 * (p1 + p2)
            eps = 1e-6  # avoid log(0)
            logm = torch.log(m + eps)

            # KL divergences
            kl1 = (p1 * (logp1 - logm)).sum(dim=1)
            kl2 = (p2 * (logp2 - logm)).sum(dim=1)

            # Jensen-Shannon Divergence
            jsd = 0.5 * (kl1 + kl2)
            return jsd
        """)
        return {"code": code}
