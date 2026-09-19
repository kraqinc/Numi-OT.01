from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any


EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"
)

SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|"
    r"private[_-]?key|password)\s*[:=]\s*\S+"
)

PROMPT_INJECTION_RE = re.compile(
    r"(?i)(ignore|disregard|forget)\s+"
    r"(all\s+)?(previous|earlier|prior)\s+instructions"
)

URL_RE = re.compile(
    r"https?://\S+",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    normalized = "\n".join(
        line
        for line in lines
        if line
    )

    return normalized.strip()


def contains_pii(text: str) -> bool:
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
    )


def contains_secret(text: str) -> bool:
    return bool(SECRET_RE.search(text))


def contains_prompt_injection(text: str) -> bool:
    return bool(PROMPT_INJECTION_RE.search(text))


def quality_score(text: str) -> float:
    if not text:
        return 0.0

    length = len(text)

    alphanumeric = sum(
        char.isalnum()
        for char in text
    )

    whitespace = sum(
        char.isspace()
        for char in text
    )

    alpha_ratio = alphanumeric / max(length, 1)
    whitespace_ratio = whitespace / max(length, 1)

    score = 0.0

    if length >= 200:
        score += 0.35

    if alpha_ratio >= 0.35:
        score += 0.35

    if whitespace_ratio <= 0.45:
        score += 0.15

    if len(set(text[:10000])) >= 20:
        score += 0.15

    return min(score, 1.0)


def document_hash(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def process_text(
    raw_text: str,
    *,
    min_characters: int = 200,
    max_characters: int = 1_000_000,
) -> tuple[str, str, float] | None:
    text = normalize_text(raw_text)

    if len(text) < min_characters:
        return None

    if len(text) > max_characters:
        return None

    if contains_pii(text):
        return None

    if contains_secret(text):
        return None

    if contains_prompt_injection(text):
        return None

    score = quality_score(text)

    if score < 0.60:
        return None

    digest = document_hash(text)

    return (
        text,
        digest,
        score,
    )


def now_iso() -> str:
    return datetime.now(
        UTC
    ).isoformat()