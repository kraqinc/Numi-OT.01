from __future__ import annotations

import torch
from torch import Tensor


def build_rope_cache(
    sequence_length: int,
    head_dim: int,
    *,
    theta: float = 1_000_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(
            "head_dim must be even for rotary embeddings."
        )

    positions = torch.arange(
        sequence_length,
        device=device,
        dtype=dtype,
    )

    frequencies = torch.arange(
        0,
        head_dim,
        2,
        device=device,
        dtype=dtype,
    )

    inverse_frequencies = 1.0 / (
        theta ** (frequencies / head_dim)
    )

    angles = torch.outer(
        positions,
        inverse_frequencies,
    )

    cos = angles.cos()
    sin = angles.sin()

    return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    rotated = torch.stack(
        (-x2, x1),
        dim=-1,
    )

    return rotated.flatten(-2)


def apply_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """
    x: [batch, heads, sequence, head_dim]
    cos/sin: [sequence, head_dim/2]
    """

    cos = torch.repeat_interleave(
        cos,
        2,
        dim=-1,
    )

    sin = torch.repeat_interleave(
        sin,
        2,
        dim=-1,
    )

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return (
        x * cos
        + rotate_half(x) * sin
    )