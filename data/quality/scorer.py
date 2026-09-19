from __future__ import annotations

import math
import re


def quality_score(text: str) -> float:
    text = text.strip()

    if not text:
        return 0.0

    length_score = min(len(text) / 2000.0, 1.0)

    words = text.split()
    lexical_score = min(len(set(words)) / max(len(words), 1), 1.0)

    sentence_count = max(len(re.findall(r"[.!?]", text)), 1)
    sentence_score = min(sentence_count / 10.0, 1.0)

    score = (
        0.40 * length_score
        + 0.35 * lexical_score
        + 0.25 * sentence_score
    )

    return float(max(0.0, min(1.0, score)))
