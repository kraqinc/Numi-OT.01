from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())


def token_accuracy(
    predictions: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> float:
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape.")

    mask = labels != ignore_index

    if not mask.any():
        return 0.0

    return float((predictions[mask] == labels[mask]).float().mean().item())


def sequence_accuracy(
    predictions: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> float:
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape.")

    mask = labels != ignore_index
    if not mask.any():
        return 0.0

    matches = (predictions == labels) | ~mask
    return float(matches.all(dim=-1).float().mean().item())


def perplexity(loss: float | Tensor) -> float:
    value = float(loss.detach().cpu().item()) if isinstance(loss, Tensor) else float(loss)

    if not math.isfinite(value):
        return float("inf")

    return math.exp(min(value, 100.0))
