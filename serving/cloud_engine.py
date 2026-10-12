"""Cloud inference via OpenAI-compatible APIs (OpenAI, xAI, DeepSeek, Groq, …).

No Anthropic. No local weights. System prompt always forced server-side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from tokenizer.templates.system import RUNTIME_SYSTEM_PROMPT, ensure_system_message


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class CloudEngine:
    """Numi product face over a remote OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.device = "cloud"

    @classmethod
    def from_env(cls) -> CloudEngine:
        base_url = os.getenv("NUMI_CLOUD_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("NUMI_CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        model = os.getenv("NUMI_CLOUD_MODEL", "gpt-4.1")
        if not api_key:
            raise RuntimeError(
                "NUMI_CLOUD_API_KEY (or OPENAI_API_KEY) is required when "
                "NUMI_BACKEND=cloud"
            )
        return cls(base_url=base_url, api_key=api_key, model=model)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        inject_system: bool = True,
        **_: Any,
    ) -> GenerationResult:
        if inject_system:
            messages = ensure_system_message(
                messages, mode="runtime", force_server_system=True
            )
        # Hard guarantee system is ours
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": RUNTIME_SYSTEM_PROMPT}, *messages]
        else:
            messages = [
                {"role": "system", "content": RUNTIME_SYSTEM_PROMPT},
                *[m for m in messages if m.get("role") != "system"],
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Cloud provider error {resp.status_code}: {resp.text[:400]}"
                )
            data = resp.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()
        usage = data.get("usage") or {}
        return GenerationResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
