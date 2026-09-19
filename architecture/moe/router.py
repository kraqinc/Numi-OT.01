from __future__ import annotations

import torch
from torch import Tensor, nn


class NumiRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        jitter_noise: float = 0.0,
    ) -> None:
        super().__init__()

        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.gate(hidden_states)

        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        probabilities = torch.softmax(logits, dim=-1)

        weights, indices = torch.topk(
            probabilities,
            self.top_k,
            dim=-1,
        )

        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return weights, indices
