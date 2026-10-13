"""Inference helpers for Numi.

Production serving uses serving.base_engine.BaseLoRAEngine (PEFT) or
serving.engine.NumiEngine (toy MoE). This module stays import-safe for CI
and does not load Qwen.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from tokenizer.templates.chat import NumiMessage
from tokenizer.templates.system import DEFAULT_NUMI_SYSTEM_PROMPT, ensure_system_message


class GenerationResult(NamedTuple):
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class NumiInferenceEngine:
    """Thin wrapper around the production PEFT engine when available.

    Importing this class never pulls Qwen. Calling from_pretrained requires
    transformers/peft and a real base+adapter.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        **_kwargs: Any,
    ) -> NumiInferenceEngine:
        from serving.base_engine import BaseLoRAEngine

        if not adapter_path:
            raise ValueError(
                "adapter_path is required (Numi PEFT LoRA). "
                "Qwen bases are disabled for product identity."
            )
        backend = BaseLoRAEngine.from_adapter(
            base_model_id=model_name,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
        )
        return cls(backend)

    def _normalize_messages(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, NumiMessage):
                normalized.append(
                    {"role": message.role, "content": message.content}
                )
                continue
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str):
                raise TypeError("Message role must be a string.")
            if not isinstance(content, str):
                raise TypeError("Message content must be a string.")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("Conversation must contain at least one message.")
        return ensure_system_message(normalized, mode="runtime", force_server_system=True)

    def generate(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        generation = config or GenerationConfig()
        normalized = self._normalize_messages(messages)
        result = self._backend.generate(
            normalized,
            max_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
        )
        return GenerationResult(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
