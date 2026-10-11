from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from architecture.positional.rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Used for Grouped-Query Attention.
    """
    if n_rep == 1:
        return hidden_states

    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):
    """
    Multi-head attention with Grouped-Query Attention (GQA) + RoPE.

    Design based on Llama 3 / Mistral attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        rope_theta: float = 500_000.0,
        max_position_embeddings: int = 8192,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        if self.head_dim * num_heads != hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"({hidden_size} vs {num_heads})"
            )

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Cache for RoPE frequencies (built on first forward)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _build_rope_cache(self, seq_len: int, device: torch.device) -> None:
        cos, sin = precompute_freqs_cis(
            dim=self.head_dim,
            end=seq_len,
            theta=self.rope_theta,
            device=device,
        )
        # Duplicate to full head_dim for easier broadcasting
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        bsz, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if self.cos_cached is None or self.cos_cached.shape[0] < seq_len:
            self._build_rope_cache(
                max(seq_len, self.max_position_embeddings),
                hidden_states.device,
            )

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        # GQA: repeat KV heads
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, torch.finfo(attn_weights.dtype).min)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        if self.attention_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)

        return self.o_proj(attn_output)
