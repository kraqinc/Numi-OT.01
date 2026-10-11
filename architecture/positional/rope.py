from __future__ import annotations

import torch
from torch import Tensor


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500_000.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Precompute the frequency tensors for RoPE (Rotary Position Embedding).

    Same formulation used by Llama 3 / Mistral.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [seq_len, dim/2]

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """
    Apply rotary position embedding.

    x   : [batch, heads, seq, head_dim]
    cos : [seq, head_dim]
    sin : [seq, head_dim]
    """
    # Expand cos/sin to match x
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)
