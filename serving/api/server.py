from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from inference.engine import NumiInferenceEngine
from serving.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)


MODEL_NAME = os.getenv(
    "NUMI_MODEL",
    "Qwen/Qwen3-30B-A3B-Base",
)

MODEL_REVISION = os.getenv(
    "NUMI_MODEL_REVISION",
    "main",
)

TRUST_REMOTE_CODE = (
    os.getenv("NUMI_TRUST_REMOTE_CODE", "false").lower()
    == "true"
)


app = FastAPI(
    title="Numi-OT.01 API",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_engine() -> NumiInferenceEngine:
    return NumiInferenceEngine.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=TRUST_REMOTE_CODE,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.get("/v1/models")
def models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "numination",
            }
        ],
    }


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    engine = get_engine()

    requested_model = request.model or MODEL_NAME

    if requested_model != MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{requested_model}'. "
                f"Available model: '{MODEL_NAME}'."
            ),
        )

    started = time.perf_counter()

    try:
        generation = engine.generate(
            [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            config=None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model inference failed: {exc}",
        ) from exc

    elapsed = time.perf_counter() - started

    _ = elapsed

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        model=MODEL_NAME,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=generation.text,
                ),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=(
                generation.prompt_tokens
                + generation.completion_tokens
            ),
        ),
    )


@app.post("/v1/chat")
def chat(request: ChatCompletionRequest) -> dict[str, object]:
    response = chat_completions(request)

    return {
        "model": response.model,
        "response": response.choices[0].message.content,
    }
