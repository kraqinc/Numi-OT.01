from __future__ import annotations

from torch import Tensor, nn

from architecture.attention.attention import Attention
from architecture.attention.cache import KVCache
from architecture.moe.moe import MoE
from architecture.normalization.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with MoE + KV-cache support.

    Structure:
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
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[Tensor, KVCache | None]:
        # Attention sub-layer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        hidden_states = residual + hidden_states

        # MoE sub-layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value
