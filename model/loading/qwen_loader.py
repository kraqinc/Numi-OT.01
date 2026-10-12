"""REMOVED path — Numination product rule: no Qwen base / identity.

Use training/base_lora + configs/training/lora_frontier.yaml instead.
"""

from __future__ import annotations


def load_qwen(*_args, **_kwargs):  # noqa: ANN003
    raise RuntimeError(
        "Qwen loader is disabled for Numination product identity. "
        "Use scripts/training/train_base_lora.py with lora_frontier.yaml "
        "(gpt-oss / DeepSeek / Gemma / MiMo — not Qwen)."
    )
