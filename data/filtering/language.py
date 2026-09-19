from __future__ import annotations

import re


SPANISH_MARKERS = {
    " el ",
    " la ",
    " los ",
    " las ",
    " que ",
    " de ",
    " para ",
    " una ",
    " con ",
}


def detect_language(text: str) -> str:
    normalized = f" {text.lower()} "

    if any(marker in normalized for marker in SPANISH_MARKERS):
        return "es"

    if re.search(r"\b(the|and|with|from|this|that)\b", normalized):
        return "en"

    return "unknown"


def is_allowed_language(
    text: str,
    allowed: tuple[str, ...] = ("es", "en"),
) -> bool:
    return detect_language(text) in allowed
