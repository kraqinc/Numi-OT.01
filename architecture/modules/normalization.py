from __future__ import annotations

import torch
from torch import Tensor, nn


class NumiRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype

        hidden_states = hidden_states.float()

        variance = hidden_states.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        hidden_states = hidden_states * torch.rsqrt(
            variance + self.eps
        )

        return (
            self.weight * hidden_states
        ).to(input_dtype)