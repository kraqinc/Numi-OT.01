from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..positional.rope import (
    apply_rope,
    build_rope_cache,
)


def repeat_kv(
    hidden_states: Tensor,
    repetitions: int,
) -> Tensor:
    if repetitions == 1:
        return hidden_states

    batch, heads, sequence, head_dim = hidden_states.shape

    hidden_states = hidden_states[:, :, None, :, :]

    hidden_states = hidden_states.expand(
        batch,
        heads,
        repetitions,
        sequence,
        head_dim,
    )

    return hidden_states.reshape(
        batch,
        heads * repetitions,
        sequence,
        head_dim,
    )


class NumiAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        *,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 32768,
    ) -> None:
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by "
                "num_attention_heads."
            )

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads."
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = (
            hidden_size // num_attention_heads
        )

        self.num_key_value_groups = (
            num_attention_heads // num_key_value_heads
        )

        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = build_rope_cache(
            sequence_length=sequence_length + position_offset,
            head_dim=self.head_dim,
            theta=self.rope_theta,
            device=hidden_states.device,
        )

        cos = cos[position_offset:]
        sin = sin[position_offset:]

        query = apply_rope(
            query,
            cos,
            sin,
        )

        key = apply_rope(
            key,
            cos,
            sin,
        )

        key = repeat_kv(
            key,
            self.num_key_value_groups,
        )

        value = repeat_kv(
            value,
            self.num_key_value_groups,
        )

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        attention_scores = attention_scores / math.sqrt(
            self.head_dim
        )

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=hidden_states.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(attention_scores.dtype)

        output = torch.matmul(
            attention_weights,
            value,
        )

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

        return self.o_proj(output)