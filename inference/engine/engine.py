from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import NamedTuple
from typing import Any

import torch

from model.loading.qwen_loader import QwenModelBundle, load_qwen
from tokenizer.templates.chat import NumiMessage
from tokenizer.templates.system import DEFAULT_NUMI_SYSTEM_PROMPT


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
    def __init__(
        self,
        bundle: QwenModelBundle,
    ) -> None:
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.model_name = bundle.model_name

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        revision: str = "main",
        trust_remote_code: bool = False,
    ) -> NumiInferenceEngine:
        bundle = load_qwen(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        return cls(bundle)

    def _normalize_messages(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []

        for message in messages:
            if isinstance(message, NumiMessage):
                normalized.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )
                continue

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str):
                raise TypeError("Message role must be a string.")

            if not isinstance(content, str):
                raise TypeError("Message content must be a string.")

            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        if not normalized:
            raise ValueError(
                "Conversation must contain at least one message."
            )

        if not any(
            message["role"] == "system"
            for message in normalized
        ):
            normalized.insert(
                0,
                DEFAULT_NUMI_SYSTEM_PROMPT.as_message(),
            )

        return normalized

    def _prepare_inputs(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "The selected tokenizer does not provide a chat template."
            )

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = self.model.device

        return {
            key: value.to(device)
            for key, value in inputs.items()
        }

    @torch.inference_mode()
    def generate(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        generation = config or GenerationConfig()

        normalized = self._normalize_messages(messages)

        inputs = self._prepare_inputs(normalized)

        input_length = inputs["input_ids"].shape[-1]

        do_sample = (
            generation.do_sample
            and generation.temperature > 0
        )

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": generation.max_new_tokens,
            "top_p": generation.top_p,
            "do_sample": do_sample,
        }

        if do_sample:
            generation_kwargs["temperature"] = (
                generation.temperature
            )

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_ids = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return GenerationResult(
            text=response.strip(),
            prompt_tokens=input_length,
            completion_tokens=int(generated_ids.shape[-1]),
        )
