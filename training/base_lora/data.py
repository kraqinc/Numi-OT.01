"""Load Numi JSONL conversations and apply chat template of the base tokenizer."""

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
    """Force Numi system prompt; drop client system roles."""
    rest = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": RUNTIME_SYSTEM_PROMPT}, *rest]


class ChatJsonlDataset(Dataset):
    def __init__(self, paths: list[str | Path], tokenizer: Any, max_length: int) -> None:
        self.conversations = [normalize_messages(m) for m in load_jsonl_messages(paths)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        messages = self.conversations[idx]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Fallback plain join
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}: {m['content']}")
            text = "\n".join(parts)

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": encoded.get("attention_mask", [1] * len(input_ids)),
            "labels": list(input_ids),
        }
