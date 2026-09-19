from __future__ import annotations


BLOCKED_PATTERNS = (
    "private key",
    "password dump",
    "credential dump",
)


def safety_flags(text: str) -> list[str]:
    normalized = text.lower()

    return [
        pattern
        for pattern in BLOCKED_PATTERNS
        if pattern in normalized
    ]


def passes_safety_filter(text: str) -> bool:
    return not safety_flags(text)
