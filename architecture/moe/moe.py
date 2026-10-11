from __future__ import annotations

import torch
from torch import Tensor, nn

from architecture.moe.experts import ExpertBank
from architecture.moe.load_balance import load_balance_loss
from architecture.moe.router import Router


class MoE(nn.Module):
    """
    Sparse Mixture-of-Experts with sorted dispatch.

    Instead of scattering masks per expert naively, tokens assigned to each
    expert are gathered once, the expert runs on a contiguous batch, and
    results are scattered back with their routing weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        router_jitter_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=router_jitter_noise,
        )

        self.experts = ExpertBank(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
        )

        self.last_aux_loss: Tensor | None = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])  # [T, H]
        num_tokens = flat.shape[0]

        weights, indices, router_probs = self.router(flat)  # [T, K], [T, K], [T, E]

        # Expand token ids for each of the top-k slots: [T*K]
        # token 0 → slots 0..K-1, token 1 → slots K..2K-1, ...
        top_k = self.top_k
        flat_indices = indices.reshape(-1)  # [T*K]
        flat_weights = weights.reshape(-1)  # [T*K]
        token_ids = torch.arange(num_tokens, device=flat.device).unsqueeze(1)
        token_ids = token_ids.expand(-1, top_k).reshape(-1)  # [T*K]

        output = torch.zeros_like(flat)

        # Process each expert on its contiguous token subset
        for expert_id in range(self.num_experts):
            mask = flat_indices == expert_id
            if not torch.any(mask):
                continue

            selected_tokens = token_ids[mask]  # which original tokens
            selected_weights = flat_weights[mask].unsqueeze(-1)
            expert_input = flat.index_select(0, selected_tokens)

            expert_output = self.experts.forward_expert(expert_id, expert_input)
            weighted = expert_output * selected_weights

            output.index_add_(0, selected_tokens, weighted)

        self.last_aux_loss = load_balance_loss(
            router_probs, indices, self.num_experts
        )

        return output.reshape(original_shape)
