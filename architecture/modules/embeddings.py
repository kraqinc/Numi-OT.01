from __future__ import annotations

import torch
from torch import Tensor, nn


class NumiTokenEmbedding(nn.Module):
    """
    Token embedding table for Numi-OT.01.

    The embedding can be resized while preserving the existing vocabulary
    weights. Newly created rows are initialized independently.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero.")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=self.initializer_range,
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)

    def resize(
        self,
        new_vocab_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> NumiTokenEmbedding:
        if new_vocab_size <= 0:
            raise ValueError("new_vocab_size must be greater than zero.")

        old_weight = self.embedding.weight.data

        new_embedding = nn.Embedding(
            num_embeddings=new_vocab_size,
            embedding_dim=self.hidden_size,
            device=device or old_weight.device,
            dtype=dtype or old_weight.dtype,
        )

        nn.init.normal_(
            new_embedding.weight,
            mean=0.0,
            std=self.initializer_range,
        )

        rows_to_copy = min(
            old_weight.shape[0],
            new_vocab_size,
        )

        with torch.no_grad():
            new_embedding.weight[:rows_to_copy].copy_(
                old_weight[:rows_to_copy]
            )

        self.embedding = new_embedding
        self.vocab_size = new_vocab_size

        return self


class NumiLMHead(nn.Module):
    """
    Vocabulary projection used for causal language modeling.

    It can share its weight matrix with NumiTokenEmbedding.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.projection = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.projection.weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states)

    def tie_to(self, embedding: NumiTokenEmbedding) -> None:
        if embedding.hidden_size != self.hidden_size:
            raise ValueError(
                "Embedding hidden size does not match LM head hidden size."
            )

        if embedding.vocab_size != self.vocab_size:
            raise ValueError(
                "Embedding vocabulary size does not match LM head."
            )

        self.projection.weight = embedding.embedding.weight

    def resize(
        self,
        new_vocab_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> NumiLMHead:
        old_weight = self.projection.weight.data

        new_projection = nn.Linear(
            self.hidden_size,
            new_vocab_size,
            bias=False,
            device=device or old_weight.device,
            dtype=dtype or old_weight.dtype,
        )

        nn.init.normal_(
            new_projection.weight,
            mean=0.0,
            std=old_weight.std().item() if old_weight.numel() else 0.02,
        )

        rows_to_copy = min(
            old_weight.shape[0],
            new_vocab_size,
        )

        with torch.no_grad():
            new_projection.weight[:rows_to_copy].copy_(
                old_weight[:rows_to_copy]
            )

        self.projection = new_projection
        self.vocab_size = new_vocab_size

        return self