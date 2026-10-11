from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Used by Llama, Mistral, Mixtral, Gemma, etc.
    More stable and slightly faster than classic LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 for numerical stability
        input_dtype = x.dtype
        x = x.float()

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        return (self.weight * x).to(input_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
