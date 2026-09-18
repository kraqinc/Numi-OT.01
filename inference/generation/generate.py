from __future__ import annotations

from collections.abc import Iterable, Mapping

from inference.engine.engine import (
    GenerationConfig,
    NumiInferenceEngine,
)
from tokenizer.templates.chat import NumiMessage


def generate(
    engine: NumiInferenceEngine,
    messages: Iterable[Mapping[str, object] | NumiMessage],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    return engine.generate(
        messages,
        config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ),
    )
