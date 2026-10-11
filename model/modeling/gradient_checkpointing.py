"""Helpers to enable/disable activation checkpointing on Numi models."""

from __future__ import annotations

from torch import nn


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Reduce activation memory during training (more compute)."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return
    # NumiModel stores the flag on the inner trunk
    inner = getattr(model, "model", model)
    if hasattr(inner, "gradient_checkpointing"):
        inner.gradient_checkpointing = True


def disable_gradient_checkpointing(model: nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        return
    inner = getattr(model, "model", model)
    if hasattr(inner, "gradient_checkpointing"):
        inner.gradient_checkpointing = False
