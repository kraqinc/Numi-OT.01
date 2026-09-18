from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=8192,
    )
