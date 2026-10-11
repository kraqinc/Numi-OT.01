from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from architecture.attention.cache import KVCache
from architecture.positional.rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads for Grouped-Query Attention."""
    if n_rep == 1:
        return hidden_states

    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):
    """
    Multi-head attention with GQA + RoPE + KV-cache.

    Uses PyTorch SDPA (FlashAttention / memory-efficient kernels when available).
    Falls back to a manual path only if SDPA is unavailable.
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

        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _build_rope_cache(self, seq_len: int, device: torch.device) -> None:
        cos, sin = precompute_freqs_cis(
            dim=self.head_dim,
            end=seq_len,
            theta=self.rope_theta,
            device=device,
        )
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[Tensor, KVCache | None]:
        bsz, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        total_len = position_offset + seq_len
        if self.cos_cached is None or self.cos_cached.shape[0] < total_len:
            self._build_rope_cache(
                max(total_len, self.max_position_embeddings),
                hidden_states.device,
            )

        cos = self.cos_cached[position_offset:total_len]
        sin = self.sin_cached[position_offset:total_len]

        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        # KV cache
        if past_key_value is not None:
            key, value = past_key_value.update(key, value)
        elif use_cache:
            past_key_value = KVCache()
            key, value = past_key_value.update(key, value)

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        # Prefer SDPA (Flash / mem-efficient / math backends)
        dropout_p = self.attention_dropout if self.training else 0.0

        # is_causal only valid when query/key lengths match and we are not
        # using an external additive mask. For cached decode (q_len=1) SDPA
        # still works without a causal mask.
        is_causal = seq_len > 1 and attention_mask is None

        try:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        except (RuntimeError, TypeError):
            # Manual fallback
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            if seq_len > 1:
                kv_len = key.shape[2]
                causal_mask = torch.triu(
                    torch.ones(
                        seq_len, kv_len, device=hidden_states.device, dtype=torch.bool
                    ),
                    diagonal=1 + (kv_len - seq_len),
                )
                attn_weights = attn_weights.masked_fill(
                    causal_mask, torch.finfo(attn_weights.dtype).min
                )
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            if dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)

        return self.o_proj(attn_output), past_key_value if use_cache else None
