from __future__ import annotations

import re

from data.normalization.normalize import normalize_text


_URL = re.compile(r"https?://\S+")
_HTML = re.compile(r"<[^>]+>")


def clean_text(
    text: str,
    remove_urls: bool = False,
    remove_html: bool = True,
) -> str:
    if remove_html:
        text = _HTML.sub(" ", text)

    if remove_urls:
        text = _URL.sub(" ", text)

    return normalize_text(text)


def clean_record(record: dict) -> dict:
    output = dict(record)
    output["text"] = clean_text(output.get("text", ""))
    return output
