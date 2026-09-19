from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LayerKV:
    key: Tensor | None = None
    value: Tensor | None = None

    def append(self, key: Tensor, value: Tensor) -> None:
        if self.key is None:
            self.key = key
            self.value = value
        else:
            self.key = torch.cat([self.key, key], dim=-2)
            self.value = torch.cat([self.value, value], dim=-2)

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.shape[-2]

    def clear(self) -> None:
        self.key = None
        self.value = None


class KVCache:
    def __init__(self, num_layers: int) -> None:
        self.layers = [LayerKV() for _ in range(num_layers)]

    def layer(self, index: int) -> LayerKV:
        return self.layers[index]

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()

    @property
    def sequence_length(self) -> int:
        return max(
            (layer.length for layer in self.layers),
            default=0,
        )
