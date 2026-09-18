from __future__ import annotations

import torch
from torch import Tensor, nn

from .mlp import NumiMLP


class NumiMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ) -> None:
        super().__init__()

        if num_experts_per_tok > num_experts:
            raise ValueError(
                "num_experts_per_tok cannot exceed num_experts."
            )

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.router = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
        )

        self.experts = nn.ModuleList(
            [
                NumiMLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        original_shape = hidden_states.shape

        flat_states = hidden_states.reshape(
            -1,
            self.hidden_size,
        )

        router_logits = self.router(flat_states)

        routing_weights = torch.softmax(
            router_logits,
            dim=-1,
            dtype=torch.float32,
        )

        top_weights, top_indices = torch.topk(
            routing_weights,
            self.num_experts_per_tok,
            dim=-1,
        )

        top_weights = top_weights / top_weights.sum(
            dim=-1,
            keepdim=True,
        )

        output = torch.zeros_like(flat_states)

        for expert_index, expert in enumerate(self.experts):
            token_indices, slot_indices = torch.where(
                top_indices == expert_index
            )

            if token_indices.numel() == 0:
                continue

            expert_input = flat_states[token_indices]

            expert_output = expert(expert_input)

            weights = top_weights[
                token_indices,
                slot_indices,
            ].unsqueeze(-1)

            output.index_add_(
                0,
                token_indices,
                expert_output * weights,
            )

        return output.reshape(original_shape)