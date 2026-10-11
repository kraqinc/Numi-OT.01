from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KVCache:
    """
    Key-value cache for one attention layer.

    Shape of key/value: [batch, num_kv_heads, seq_len, head_dim]
    """

    key: Tensor | None = None
    value: Tensor | None = None

    def update(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        if self.key is None:
            self.key = key_states
            self.value = value_states
        else:
            self.key = torch.cat([self.key, key_states], dim=2)
            self.value = torch.cat([self.value, value_states], dim=2)

        return self.key, self.value

    def get_seq_length(self) -> int:
        if self.key is None:
            return 0
        return int(self.key.shape[2])

    def clear(self) -> None:
        self.key = None
        self.value = None
