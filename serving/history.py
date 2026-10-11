"""Trim chat history by approximate token budget before generation."""

from __future__ import annotations

from typing import Any


def estimate_tokens(text: str) -> int:
    # rough: ~4 chars/token for mixed ES/EN
    return max(1, len(text) // 4)


def trim_messages(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1500,
    keep_system: bool = True,
) -> list[dict[str, str]]:
    if not messages:
        return messages

    system = [m for m in messages if m.get("role") == "system"] if keep_system else []
    rest = [m for m in messages if m.get("role") != "system"] if keep_system else list(messages)

    budget = max_tokens
    for s in system:
        budget -= estimate_tokens(s.get("content", ""))

    kept: list[dict[str, str]] = []
    for m in reversed(rest):
        cost = estimate_tokens(m.get("content", "")) + 4
        if cost > budget and kept:
            break
        if cost <= budget:
            kept.append(m)
            budget -= cost
    kept.reverse()
    return system + kept
