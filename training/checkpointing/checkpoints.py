"""Lightweight checkpoint save/load for Numi training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: dict[str, Any] | None = None,
    step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
    lora_only: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if lora_only:
        from training.lora.lora import lora_state_dict

        model_sd = lora_state_dict(model)
    else:
        model_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    payload: dict[str, Any] = {
        "model": model_sd,
        "step": step,
        "config": config or {},
        "lora_only": lora_only,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model_sd = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(model_sd, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {
        "step": payload.get("step", 0),
        "config": payload.get("config", {}),
        "missing": missing,
        "unexpected": unexpected,
        "lora_only": payload.get("lora_only", False),
    }
