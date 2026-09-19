from __future__ import annotations

import torch
from torch import Tensor


def dispatch_tokens(
    hidden_states: Tensor,
    expert_indices: Tensor,
) -> list[Tensor]:
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_indices = expert_indices.reshape(-1)

    dispatched: list[Tensor] = []

    for expert_id in range(int(flat_indices.max().item()) + 1):
        mask = flat_indices == expert_id
        dispatched.append(flat_hidden[mask])

    return dispatched


def combine_expert_outputs(
    outputs: Tensor,
    weights: Tensor,
    indices: Tensor,
    num_tokens: int,
) -> Tensor:
    hidden_size = outputs.shape[-1]
    result = outputs.new_zeros(num_tokens, hidden_size)

    result.index_add_(
        0,
        indices,
        outputs * weights.unsqueeze(-1),
    )

    return result
