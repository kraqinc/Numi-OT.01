from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Single expert (SwiGLU FFN).

    Same structure used by Llama / Mistral / Mixtral.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertBank(nn.Module):
    """Collection of independent experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward_expert(self, expert_idx: int, x: Tensor) -> Tensor:
        return self.experts[expert_idx](x)
