from __future__ import annotations

import torch
from torch import Tensor, nn

from .experts import NumiExpertBank
from .load_balance import load_balance_loss
from .router import NumiRouter


class NumiMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        router_jitter: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = NumiRouter(
            hidden_size,
            num_experts,
            top_k,
            router_jitter,
        )

        self.experts = NumiExpertBank(
            hidden_size,
            intermediate_size,
            num_experts,
        )

        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])

        weights, indices = self.router(flat)

        output = torch.zeros_like(flat)

        for expert_id in range(self.num_experts):
            token_positions, top_positions = torch.where(indices == expert_id)

            if token_positions.numel() == 0:
                continue

            selected = flat[token_positions]
            expert_output = self.experts.forward_expert(
                expert_id,
                selected,
            )

            expert_weights = weights[
                token_positions,
                top_positions,
            ]

            output.index_add_(
                0,
                token_positions,
                expert_output * expert_weights.unsqueeze(-1),
            )

        router_probs = self.router.gate(flat).softmax(dim=-1)

        self.last_aux_loss = load_balance_loss(
            router_probs,
            indices,
            self.num_experts,
        )

        return output.reshape(original_shape)
