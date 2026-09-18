from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


_ALLOWED_ROLES = {
    "system",
    "user",
    "assistant",
    "tool",
}


@dataclass(frozen=True)
class NumiMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ALLOWED_ROLES:
            raise ValueError(
                f"Invalid role '{self.role}'. "
                f"Expected one of: {sorted(_ALLOWED_ROLES)}"
            )

        if not isinstance(self.content, str):
            raise TypeError("Message content must be a string.")


class NumiChatTemplate:
    def __init__(
        self,
        *,
        bos_token: str,
        eos_token: str,
        system_token: str,
        user_token: str,
        assistant_token: str,
        tool_token: str,
    ) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.role_tokens = {
            "system": system_token,
            "user": user_token,
            "assistant": assistant_token,
            "tool": tool_token,
        }

    @classmethod
    def from_special_tokens(cls, tokens: object) -> NumiChatTemplate:
        return cls(
            bos_token=tokens.bos,
            eos_token=tokens.eos,
            system_token=tokens.system,
            user_token=tokens.user,
            assistant_token=tokens.assistant,
            tool_token=tokens.tool,
        )

    def normalize_messages(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
    ) -> list[NumiMessage]:
        normalized: list[NumiMessage] = []

        for message in messages:
            if isinstance(message, NumiMessage):
                normalized.append(message)
                continue

            if not isinstance(message, Mapping):
                raise TypeError(
                    "Each message must be a mapping or NumiMessage."
                )

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str):
                raise TypeError("Message role must be a string.")

            if not isinstance(content, str):
                raise TypeError("Message content must be a string.")

            normalized.append(
                NumiMessage(
                    role=role,
                    content=content,
                )
            )

        if not normalized:
            raise ValueError("Conversation must contain at least one message.")

        return normalized

    def render(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        normalized = self.normalize_messages(messages)

        parts: list[str] = [self.bos_token]

        for message in normalized:
            role_token = self.role_tokens[message.role]
            content = message.content.strip()

            parts.append(role_token)

            if content:
                parts.append(content)

            parts.append(self.eos_token)

        if add_generation_prompt:
            parts.append(self.role_tokens["assistant"])

        return "\n".join(parts) + "\n"