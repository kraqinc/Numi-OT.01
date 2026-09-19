from __future__ import annotations

import re
import unicodedata


_WHITESPACE = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()


def normalize_record(record: dict) -> dict:
    output = dict(record)
    output["text"] = normalize_text(str(output.get("text", "")))
    return output
