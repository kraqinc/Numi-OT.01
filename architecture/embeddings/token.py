from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Token embedding table with resize support."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        initializer_range: float = 0.02,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.initializer_range)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)

    def resize(self, new_vocab_size: int) -> TokenEmbedding:
        old_weight = self.embedding.weight.data
        new_emb = nn.Embedding(
            new_vocab_size,
            self.hidden_size,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        nn.init.normal_(new_emb.weight, mean=0.0, std=self.initializer_range)

        n = min(old_weight.shape[0], new_vocab_size)
        with torch.no_grad():
            new_emb.weight[:n].copy_(old_weight[:n])

        self.embedding = new_emb
        self.vocab_size = new_vocab_size
        return self


class LMHead(nn.Module):
    """Language modeling head (can share weights with TokenEmbedding)."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def weight(self) -> nn.Parameter:
        return self.projection.weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states)

    def tie_to(self, embedding: TokenEmbedding) -> None:
        if embedding.hidden_size != self.hidden_size:
            raise ValueError("Hidden size mismatch when tying embeddings")
        if embedding.vocab_size != self.vocab_size:
            raise ValueError("Vocab size mismatch when tying embeddings")
        self.projection.weight = embedding.embedding.weight

    def resize(self, new_vocab_size: int) -> LMHead:
        old_weight = self.projection.weight.data
        new_proj = nn.Linear(
            self.hidden_size,
            new_vocab_size,
            bias=False,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        nn.init.normal_(new_proj.weight, mean=0.0, std=0.02)

        n = min(old_weight.shape[0], new_vocab_size)
        with torch.no_grad():
            new_proj.weight[:n].copy_(old_weight[:n])

        self.projection = new_proj
        self.vocab_size = new_vocab_size
        return self
