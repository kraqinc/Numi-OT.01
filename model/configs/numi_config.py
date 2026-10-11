from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class NumiConfig(PretrainedConfig):
    """
    Configuration for Numi-OT.01

    Clean design inspired by Llama 3 (dense backbone) + Mixtral (MoE routing).
    No external model inheritance.
    """

    model_type = "numi"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        moe_intermediate_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500_000.0,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        router_jitter_noise: float = 0.0,
        aux_loss_coef: float = 0.01,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.router_jitter_noise = router_jitter_noise
        self.aux_loss_coef = aux_loss_coef

        # validations
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed "
                f"num_experts ({num_experts})"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["model_type"] = self.model_type
        return output
