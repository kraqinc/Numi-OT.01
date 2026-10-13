"""Qwen path permanently disabled — Numination product rule."""

from __future__ import annotations

from typing import Any


class QwenModelBundle:  # kept name only so old imports fail loudly at call time
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Qwen is disabled for Numination. Use PEFT (SmolLM2 + Numi LoRA)."
        )


def load_qwen(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "Qwen loader is disabled for Numination product identity. "
        "Use NUMI_BACKEND=peft with NUMI_PEFT_BASE + NUMI_PEFT_ADAPTER."
    )
