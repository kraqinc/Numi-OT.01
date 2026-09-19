from __future__ import annotations

import torch
from torch import Tensor


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    return logits / temperature


def top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.shape[-1])
    values, _ = torch.topk(logits, top_k, dim=-1)

    threshold = values[..., -1, None]

    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)

    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    filtered = sorted_logits.masked_fill(
        remove,
        float("-inf"),
    )

    return torch.full_like(logits, float("-inf")).scatter(
        -1,
        sorted_indices,
        filtered,
    )


def sample_next_token(
    logits: Tensor,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Tensor:
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probabilities = torch.softmax(logits, dim=-1)

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )
