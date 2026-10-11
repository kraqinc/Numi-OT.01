from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from architecture.attention.cache import KVCache
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
    past_key_values: list[KVCache] | None = None


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
    Core Numi transformer with full KV-cache support.

    Architecture inspired by Llama 3 (attention/RoPE/RMSNorm)
    and Mixtral (sparse MoE). Built from scratch.
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
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[KVCache] | None]:
        hidden_states = self.embed_tokens(input_ids)

        # Position offset for RoPE when using cache
        position_offset = 0
        if past_key_values is not None and len(past_key_values) > 0:
            position_offset = past_key_values[0].get_seq_length()

        present_key_values: list[KVCache] | None = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Checkpointing is incompatible with use_cache during training
                hidden_states, _ = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    None,
                    False,
                    position_offset,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past,
                    use_cache=use_cache,
                    position_offset=position_offset,
                )
                if use_cache and present_key_values is not None:
                    present_key_values.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, present_key_values


class NumiForCausalLM(NumiPreTrainedModel):
    """Causal LM head with KV-cache and simple generation support."""

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
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        hidden_states, present_key_values = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

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

        # MoE load-balancing loss
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
            past_key_values=present_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Simple autoregressive generation with KV-cache.

        Args:
            input_ids:       [batch, seq]
            max_new_tokens:  how many tokens to generate
            temperature:     sampling temperature (1.0 = no change)
            top_k:           keep only top-k logits (optional)
            top_p:           nucleus sampling (optional)
            eos_token_id:    stop when this token is generated

        Returns:
            generated token ids [batch, seq + new_tokens]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids
        past_key_values: list[KVCache] | None = None

        for _ in range(max_new_tokens):
            # On first step use full sequence, afterwards only the last token
            if past_key_values is None:
                model_input = generated
            else:
                model_input = generated[:, -1:]

            outputs = self.forward(
                model_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]  # [batch, vocab]
            past_key_values = outputs.past_key_values

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # Top-p (nucleus)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return generated
