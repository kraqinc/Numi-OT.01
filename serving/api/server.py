from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from serving.engine import NumiEngine
from serving.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)

MODEL_ID = os.getenv("NUMI_MODEL_ID", "numi-ot-0.1")
CHECKPOINT = os.getenv("NUMI_CHECKPOINT")  # optional path to final.pt
CONFIG_PATH = os.getenv("NUMI_CONFIG", "configs/training/sft.yaml")
DEVICE = os.getenv("NUMI_DEVICE", "auto")

app = FastAPI(
    title="Numi-OT.01 API",
    version="0.1.0",
    description="OpenAI-compatible chat API for Numi",
)


@lru_cache(maxsize=1)
def get_engine() -> NumiEngine:
    return NumiEngine.from_checkpoint(
        checkpoint=CHECKPOINT,
        config_path=CONFIG_PATH,
        device=DEVICE,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "numination",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    engine = get_engine()

    requested = request.model or MODEL_ID
    if requested not in {MODEL_ID, "numi", "numi-ot-0.1", "Numi-OT.01"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{requested}'. Available: '{MODEL_ID}'.",
        )

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    started = time.perf_counter()
    try:
        generation = engine.generate(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Inference failed: {exc}") from exc

    _ = time.perf_counter() - started

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        model=MODEL_ID,
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
            total_tokens=generation.prompt_tokens + generation.completion_tokens,
        ),
    )


@app.post("/v1/chat")
def chat_simple(request: ChatCompletionRequest) -> dict[str, object]:
    response = chat_completions(request)
    return {
        "model": response.model,
        "response": response.choices[0].message.content,
    }
