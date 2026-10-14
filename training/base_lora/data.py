"""Load Numi JSONL conversations; train loss only on assistant turns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from tokenizer.templates.system import RUNTIME_SYSTEM_PROMPT


def load_jsonl_messages(paths: list[str | Path]) -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    rows.append(msgs)
    if not rows:
        raise FileNotFoundError(f"No conversations found in {paths}")
    return rows


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    rest = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": RUNTIME_SYSTEM_PROMPT}, *rest]


def _tokenize_text(tokenizer: Any, text: str, max_length: int) -> list[int]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    return list(encoded["input_ids"])


def _build_masked_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
) -> dict[str, Any]:
    """Full conversation tokens; labels = -100 except assistant spans."""
    # Prefer chat template for the full turn (training target includes assistant)
    if hasattr(tokenizer, "apply_chat_template"):
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Prompt up to (not including) last assistant content: rebuild without last assistant
        if messages and messages[-1].get("role") == "assistant":
            prefix_msgs = messages[:-1]
            prefix_text = tokenizer.apply_chat_template(
                prefix_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prefix_text = full_text
    else:
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        full_text = "\n".join(parts)
        if messages and messages[-1].get("role") == "assistant":
            prefix_parts = [
                f"{m['role'].upper()}: {m['content']}" for m in messages[:-1]
            ]
            prefix_text = "\n".join(prefix_parts) + "\nASSISTANT:"
        else:
            prefix_text = full_text

    full_ids = _tokenize_text(tokenizer, full_text, max_length)
    prefix_ids = _tokenize_text(tokenizer, prefix_text, max_length)
    prefix_len = min(len(prefix_ids), len(full_ids))

    labels = [-100] * prefix_len + full_ids[prefix_len:]
    # Align lengths
    if len(labels) < len(full_ids):
        labels.extend([-100] * (len(full_ids) - len(labels)))
    labels = labels[: len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class ChatJsonlDataset(Dataset):
    def __init__(self, paths: list[str | Path], tokenizer: Any, max_length: int) -> None:
        self.conversations = [normalize_messages(m) for m in load_jsonl_messages(paths)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return _build_masked_example(
            self.tokenizer, self.conversations[idx], self.max_length
        )
