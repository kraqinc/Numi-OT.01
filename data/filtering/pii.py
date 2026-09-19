from __future__ import annotations

import re


EMAIL = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

PHONE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")


def find_pii(text: str) -> dict[str, list[str]]:
    return {
        "emails": EMAIL.findall(text),
        "phones": PHONE.findall(text),
    }


def contains_pii(text: str) -> bool:
    result = find_pii(text)
    return bool(result["emails"] or result["phones"])


def redact_pii(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    return PHONE.sub("[PHONE]", text)
