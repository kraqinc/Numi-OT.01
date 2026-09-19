from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator


def text_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_records(records: Iterable[dict]) -> Iterator[dict]:
    seen: set[str] = set()

    for record in records:
        fingerprint = text_fingerprint(record.get("text", ""))

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        yield record


def deduplicate_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for text in texts:
        fingerprint = text_fingerprint(text)

        if fingerprint not in seen:
            seen.add(fingerprint)
            result.append(text)

    return result
