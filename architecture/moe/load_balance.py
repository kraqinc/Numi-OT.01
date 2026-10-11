from __future__ import annotations

import torch
from torch import Tensor


def load_balance_loss(
    router_probs: Tensor,
    expert_indices: Tensor,
    num_experts: int,
) -> Tensor:
    """
    Switch Transformer style load balancing loss.

    Encourages the router to distribute tokens evenly across experts.

    Args:
        router_probs   : [num_tokens, num_experts]  (full softmax)
        expert_indices : [num_tokens, top_k]
        num_experts    : int
    """
    # Fraction of tokens routed to each expert
    num_tokens = expert_indices.shape[0]
    expert_mask = torch.zeros(
        num_tokens, num_experts, device=expert_indices.device, dtype=torch.float32
    )
    expert_mask.scatter_(1, expert_indices, 1.0)

    tokens_per_expert = expert_mask.mean(dim=0)          # [num_experts]
    router_prob_per_expert = router_probs.mean(dim=0)    # [num_experts]

    # The classic formulation from Switch Transformer / Mixtral
    loss = num_experts * (tokens_per_expert * router_prob_per_expert).sum()
    return loss
