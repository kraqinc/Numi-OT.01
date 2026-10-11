from __future__ import annotations

from torch import Tensor, nn

from architecture.attention.attention import Attention
from architecture.moe.moe import MoE
from architecture.normalization.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with MoE.

    Structure (Llama / Mixtral style):

        x → RMSNorm → Attention → + residual
          → RMSNorm → MoE       → + residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        rope_theta: float = 500_000.0,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 8192,
        attention_dropout: float = 0.0,
        router_jitter_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            attention_dropout=attention_dropout,
        )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.mlp = MoE(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            router_jitter_noise=router_jitter_noise,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Attention sub-layer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MoE sub-layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
