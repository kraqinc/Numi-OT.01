from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Router(nn.Module):
    """
    Top-k expert router (Mixtral style).

    Returns:
        weights     : [num_tokens, top_k]  renormalized
        indices     : [num_tokens, top_k]
        router_probs: [num_tokens, num_experts]  full softmax (for aux loss)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        jitter_noise: float = 0.0,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot be larger than num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.gate(hidden_states)

        if self.training and self.jitter_noise > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_noise

        router_probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)

        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return weights, indices, router_probs
