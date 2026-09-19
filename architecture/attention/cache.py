from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KVCache:
    key: Tensor | None = None
    value: Tensor | None = None

    @property
    def seq_len(self) -> int:
        if self.key is None:
            return 0
        return self.key.shape[-2]

    def update(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if self.key is None:
            self.key = key
            self.value = value
        else:
            self.key = torch.cat((self.key, key), dim=-2)
            self.value = torch.cat((self.value, value), dim=-2)

        return self.key, self.value

    def clear(self) -> None:
        self.key = None
        self.value = None

    def to(self, device: torch.device | str) -> "KVCache":
        if self.key is not None:
            self.key = self.key.to(device)
        if self.value is not None:
            self.value = self.value.to(device)
        return self


class KVCacheManager:
    def __init__(self, num_layers: int) -> None:
        self.layers = [KVCache() for _ in range(num_layers)]

    def __getitem__(self, layer_idx: int) -> KVCache:
        return self.layers[layer_idx]

    def clear(self) -> None:
        for cache in self.layers:
            cache.clear()

    @property
    def seq_len(self) -> int:
        for cache in self.layers:
            if cache.seq_len:
                return cache.seq_len
        return 0
