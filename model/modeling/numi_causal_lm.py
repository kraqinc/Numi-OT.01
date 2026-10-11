from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from architecture.blocks.transformer import NumiTransformerBlock
from architecture.embeddings.token import NumiLMHead, NumiTokenEmbedding
from architecture.normalization.rmsnorm import NumiRMSNorm
from model.configs.numi_config import NumiConfig


@dataclass
class NumiCausalLMOutput:
    loss: Tensor | None = None
    logits: Tensor | None = None
    hidden_states: Tensor | None = None
    aux_loss: Tensor | None = None


class NumiPreTrainedModel(PreTrainedModel):
    config_class = NumiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NumiTransformerBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class NumiModel(NumiPreTrainedModel):
    """
    Core Numi transformer.

    Layout (following Qwen2-MoE / Llama / Mixtral convention):

        input_ids
            ↓
        Token Embedding
            ↓
        [NumiTransformerBlock] × num_hidden_layers
            ↓
        Final RMSNorm
    """

    def __init__(self, config: NumiConfig) -> None:
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = NumiTokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
        )

        self.layers = nn.ModuleList(
            [
                NumiTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    moe_intermediate_size=config.moe_intermediate_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    max_position_embeddings=config.max_position_embeddings,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = NumiRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, NumiTokenEmbedding):
            raise TypeError("NumiModel requires a NumiTokenEmbedding.")
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_offset,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_offset=position_offset,
                )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class NumiForCausalLM(NumiPreTrainedModel):
    """
    Causal LM head on top of NumiModel.

    Same interface style as HuggingFace CausalLM models.
    """

    _tied_weights_keys = ["lm_head.projection.weight"]

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
            raise TypeError("Numi LM head requires an nn.Linear projection.")
        self.lm_head.projection = new_embeddings

    def tie_weights(self) -> None:
        if not self.config.tie_word_embeddings:
            return

        embedding = self.model.embed_tokens
        self.lm_head.vocab_size = embedding.vocab_size
        self.lm_head.hidden_size = embedding.hidden_size
        self.lm_head.tie_to(embedding)

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
    ) -> NumiTokenEmbedding:
        if new_num_tokens <= 0:
            raise ValueError("new_num_tokens must be greater than zero.")

        old_vocab_size = self.config.vocab_size

        self.model.embed_tokens.resize(new_num_tokens)
        self.lm_head.vocab_size = old_vocab_size
        self.lm_head.resize(new_num_tokens)

        self.config.vocab_size = new_num_tokens

        if self.config.tie_word_embeddings:
            self.tie_weights()

        return self.model.embed_tokens

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        position_offset: int = 0,
    ) -> NumiCausalLMOutput:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_offset=position_offset,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Collect MoE auxiliary loss from all layers (load balancing)
        aux_loss = None
        for layer in self.model.layers:
            if hasattr(layer.moe, "last_aux_loss"):
                if aux_loss is None:
                    aux_loss = layer.moe.last_aux_loss
                else:
                    aux_loss = aux_loss + layer.moe.last_aux_loss

        if aux_loss is not None and self.training:
            # Standard scale used in Mixtral / Qwen2-MoE
            aux_loss = aux_loss * 0.001
            if loss is not None:
                loss = loss + aux_loss

        return NumiCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            aux_loss=aux_loss,
        )
