from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from .configuration_numi import NumiConfig
from .modules.embeddings import NumiLMHead, NumiTokenEmbedding


@dataclass
class NumiCausalLMOutput:
    loss: Tensor | None = None
    logits: Tensor | None = None


class NumiPreTrainedModel(PreTrainedModel):
    config_class = NumiConfig
    base_model_prefix = "numi"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )


class NumiModel(NumiPreTrainedModel):
    """
    Core Numi model container.

    Transformer blocks are deliberately kept as a separate layer from the
    vocabulary system so vocabulary changes do not require rewriting the
    entire model.
    """

    def __init__(self, config: NumiConfig) -> None:
        super().__init__(config)

        self.embeddings = NumiTokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
        )

        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, NumiTokenEmbedding):
            raise TypeError(
                "NumiModel requires a NumiTokenEmbedding."
            )

        self.embeddings = value

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
    ) -> NumiTokenEmbedding:
        if new_num_tokens <= 0:
            raise ValueError(
                "new_num_tokens must be greater than zero."
            )

        self.embeddings.resize(new_num_tokens)

        self.config.vocab_size = new_num_tokens

        return self.embeddings

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        hidden_states = self.embeddings(input_ids)

        return self.norm(hidden_states)


class NumiForCausalLM(NumiPreTrainedModel):
    """
    Causal language model wrapper.

    The transformer stack will live between NumiModel and NumiLMHead.
    This version establishes the vocabulary/embedding contract first.
    """

    _tied_weights_keys = [
        "lm_head.projection.weight",
    ]

    def __init__(self, config: NumiConfig) -> None:
        super().__init__(config)

        self.model = NumiModel(config)

        self.lm_head = NumiLMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )

        if config.tie_word_embeddings:
            self.tie_weights()

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head.projection

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if not isinstance(new_embeddings, nn.Linear):
            raise TypeError(
                "Numi LM head requires an nn.Linear projection."
            )

        self.lm_head.projection = new_embeddings

    def tie_weights(self) -> None:
        if not self.config.tie_word_embeddings:
            return

        embedding = self.model.embeddings

        self.lm_head.vocab_size = embedding.vocab_size
        self.lm_head.hidden_size = embedding.hidden_size
        self.lm_head.tie_to(embedding)

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
    ) -> NumiTokenEmbedding:
        if new_num_tokens <= 0:
            raise ValueError(
                "new_num_tokens must be greater than zero."
            )

        old_vocab_size = self.config.vocab_size

        self.model.embeddings.resize(new_num_tokens)

        self.lm_head.vocab_size = old_vocab_size
        self.lm_head.resize(new_num_tokens)

        self.config.vocab_size = new_num_tokens

        if self.config.tie_word_embeddings:
            self.tie_weights()

        return self.model.embeddings

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> NumiCausalLMOutput:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return NumiCausalLMOutput(
            loss=loss,
            logits=logits,
        )