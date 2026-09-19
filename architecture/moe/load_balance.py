from __future__ import annotations

import torch
from torch import Tensor


def load_balance_loss(
    router_probs: Tensor,
    expert_indices: Tensor,
    num_experts: int,
) -> Tensor:
    probs = router_probs.reshape(-1, num_experts)
    indices = expert_indices.reshape(-1)

    importance = probs.mean(dim=0)

    load = torch.zeros(
        num_experts,
        device=probs.device,
        dtype=probs.dtype,
    )

    load.scatter_add_(
        0,
        indices,
        torch.ones_like(indices, dtype=probs.dtype),
    )

    load = load / max(indices.numel(), 1)

    return num_experts * torch.sum(importance * load)
