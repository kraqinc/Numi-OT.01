from __future__ import annotations

import torch
from torch import Tensor, nn


class NumiExpert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class NumiExpertBank(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ) -> None:
        super().__init__()

        self.experts = nn.ModuleList(
            [
                NumiExpert(hidden_size, intermediate_size)
                for _ in range(num_experts)
            ]
        )

    def forward_expert(
        self,
        expert_id: int,
        hidden_states: Tensor,
    ) -> Tensor:
        return self.experts[expert_id](hidden_states)
