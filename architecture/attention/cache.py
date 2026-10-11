from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class KVCache:
    """
    Simple key-value cache for one attention layer.

    Stores the accumulated key and value tensors across generation steps.
    Shape of key/value: [batch, num_kv_heads, seq_len, head_dim]
    """

    key: Tensor | None = None
    value: Tensor | None = None

    def update(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """
        Append new key/value states and return the full cache.

        Args:
            key_states:   [batch, num_kv_heads, new_seq, head_dim]
            value_states: [batch, num_kv_heads, new_seq, head_dim]

        Returns:
            (full_key, full_value) ready for attention.
        """
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
        return self.key.shape[2]

    def clear(self) -> None:
        self.key = None
        self.value = None


# Need torch for cat
import torch  # noqa: E402
