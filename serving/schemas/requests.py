from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ClientRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: ClientRole
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=512)
    stream: bool = False
