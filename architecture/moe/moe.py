from __future__ import annotations

import torch
from torch import Tensor, nn

from architecture.moe.experts import ExpertBank
from architecture.moe.load_balance import load_balance_loss
from architecture.moe.router import Router


class MoE(nn.Module):
    """
    Sparse Mixture-of-Experts layer (Mixtral style).

    - Top-k routing
    - Independent SwiGLU experts
    - Load-balancing auxiliary loss
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
        # Flatten to [num_tokens, hidden]
        flat = hidden_states.reshape(-1, original_shape[-1])

        weights, indices = self.router(flat)  # [T, k], [T, k]

        # Dispatch
        output = torch.zeros_like(flat)

        for expert_id in range(self.num_experts):
            # Find which tokens selected this expert
            token_mask = (indices == expert_id).any(dim=-1)
            if not token_mask.any():
                continue

            token_idx = token_mask.nonzero(as_tuple=True)[0]
            selected = flat[token_idx]

            expert_out = self.experts.forward_expert(expert_id, selected)

            # Gather the weight that corresponds to this expert for each token
            # indices: [T, k] → find the position of expert_id in the top-k
            expert_pos = (indices[token_idx] == expert_id).float().argmax(dim=-1)
            expert_weight = weights[token_idx, expert_pos].unsqueeze(-1)

            output.index_add_(0, token_idx, expert_out * expert_weight)

        # Auxiliary load-balancing loss
        with torch.no_grad():
            full_probs = self.router.gate(flat).softmax(dim=-1)

        self.last_aux_loss = load_balance_loss(
            full_probs, indices, self.num_experts
        )

        return output.reshape(original_shape)
