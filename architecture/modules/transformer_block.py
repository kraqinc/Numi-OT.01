from __future__ import annotations

from torch import Tensor, nn

from .attention import NumiAttention
from .moe import NumiMoE
from .normalization import NumiRMSNorm


class NumiTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
    ) -> None:
        super().__init__()

        self.input_layernorm = NumiRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
        )

        self.self_attn = NumiAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )

        self.post_attention_layernorm = NumiRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
        )

        self.moe = NumiMoE(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(
            hidden_states
        )

        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(
            hidden_states
        )

        hidden_states = self.moe(
            hidden_states
        )

        hidden_states = residual + hidden_states

        return hidden_states