"""Minimal LoRA implementation for Numi (no peft dependency)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "projection",  # LM head linear name in Numi
    )


class LoRALinear(nn.Module):
    """Wrap an existing nn.Linear with low-rank adapters."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        result = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return result + lora_out * self.scaling

    def merge(self) -> None:
        """Fold LoRA weights into the base linear (in-place)."""
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.add_(delta)
            nn.init.zeros_(self.lora_B)


def _match_target(name: str, targets: Iterable[str]) -> bool:
    return any(t in name for t in targets)


def apply_lora(model: nn.Module, config: LoRAConfig) -> int:
    """Replace matching Linear modules with LoRALinear. Returns count."""
    replaced = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _match_target(full, config.target_modules):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, r=config.r, alpha=config.alpha, dropout=config.dropout),
                )
                replaced += 1
    return replaced


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora(model: nn.Module) -> int:
    """Merge all LoRALinear layers into base weights. Returns count."""
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            n += 1
    return n
