from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor

from model.modeling.numi_causal_lm import NumiForCausalLM


@torch.no_grad()
def sequence_nll(
    model: NumiForCausalLM,
    input_ids: Tensor,
) -> tuple[float, int]:
    """Negative log-likelihood sum and token count (excluding first token)."""
    model.eval()
    outputs = model(input_ids=input_ids, labels=input_ids)
    # loss is mean NLL; recover sum approximately via * (S-1)
    seq_len = input_ids.shape[1]
    n_tokens = max(seq_len - 1, 1)
    assert outputs.loss is not None
    return float(outputs.loss.item() * n_tokens), n_tokens


@torch.no_grad()
def corpus_perplexity(
    model: NumiForCausalLM,
    batches: Iterable[Tensor],
) -> float:
    total_nll = 0.0
    total_tokens = 0
    for batch in batches:
        nll, n = sequence_nll(model, batch)
        total_nll += nll
        total_tokens += n
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)
