from __future__ import annotations

import torch
from torch import Tensor, nn


class NumiMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        gated = torch.nn.functional.silu(
            self.gate_proj(hidden_states)
        )

        return self.down_proj(
            gated * self.up_proj(hidden_states)
        )