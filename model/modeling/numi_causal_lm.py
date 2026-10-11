from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from architecture.blocks.transformer import TransformerBlock
from architecture.embeddings.token import LMHead, TokenEmbedding
from architecture.normalization.rmsnorm import RMSNorm
from model.configs.numi_config import NumiConfig


@dataclass
class CausalLMOutput:
    loss: Tensor | None = None
    logits: Tensor | None = None
    hidden_states: Tensor | None = None
    aux_loss: Tensor | None = None


class NumiPreTrainedModel(PreTrainedModel):
    config_class = NumiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class NumiModel(NumiPreTrainedModel):
    """
    Core Numi transformer (from-scratch design).

    Architecture inspired by:
    - Llama 3   → attention, RoPE, RMSNorm, residual style
    - Mixtral   → sparse Mixture-of-Experts

    No code or weights inherited from any existing model.
    """

    def __init__(self, config: NumiConfig) -> None:
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
            padding_idx=config.pad_token_id,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    moe_intermediate_size=config.moe_intermediate_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    max_position_embeddings=config.max_position_embeddings,
                    attention_dropout=config.attention_dropout,
                    router_jitter_noise=config.router_jitter_noise,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)

        return self.norm(hidden_states)


class NumiForCausalLM(NumiPreTrainedModel):
    """Causal language model head on top of NumiModel."""

    _tied_weights_keys = ["lm_head.projection.weight"]

    def __init__(self, config: NumiConfig) -> None:
        super().__init__(config)

        self.model = NumiModel(config)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size)

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
        self.lm_head.projection = new_embeddings

    def tie_weights(self) -> None:
        if not self.config.tie_word_embeddings:
            return
        emb = self.model.embed_tokens
        self.lm_head.vocab_size = emb.vocab_size
        self.lm_head.hidden_size = emb.hidden_size
        self.lm_head.tie_to(emb)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Collect MoE load-balancing loss
        aux_loss = None
        for layer in self.model.layers:
            if layer.mlp.last_aux_loss is not None:
                if aux_loss is None:
                    aux_loss = layer.mlp.last_aux_loss
                else:
                    aux_loss = aux_loss + layer.mlp.last_aux_loss

        if aux_loss is not None and self.training:
            aux_loss = aux_loss * self.config.aux_loss_coef
            if loss is not None:
                loss = loss + aux_loss

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            aux_loss=aux_loss,
        )
