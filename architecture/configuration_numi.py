from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class NumiConfig(PretrainedConfig):
    model_type = "numi"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        intermediate_size: int = 6144,
        moe_intermediate_size: int = 768,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        initializer_range: float = 0.02,
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

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        self.initializer_range = initializer_range

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads."
            )

        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                "num_experts_per_tok cannot exceed num_experts."
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> NumiConfig:
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        **kwargs: Any,
    ) -> NumiConfig:
        """
        Build a model configuration using the actual tokenizer vocabulary.

        This is important because Numi adds its own special tokens.
        """
        return cls(
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

    def update_vocab_size(self, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")

        self.vocab_size = int(vocab_size)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()

        output["model_type"] = self.model_type

        return output