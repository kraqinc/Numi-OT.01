from __future__ import annotations

import json
import logging
import os
import time
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from serving.concurrency import inference_slot
from serving.engine import NumiEngine
from serving.history import trim_messages
from serving.metrics import METRICS
from serving.middleware.auth import APIKeyMiddleware
from serving.middleware.logging import RequestLoggingMiddleware
from serving.middleware.rate_limit import RateLimitMiddleware
from serving.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)

logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("NUMI_MODEL_ID", "numi-ot-0.1")
CHECKPOINT = os.getenv("NUMI_CHECKPOINT")
CONFIG_PATH = os.getenv("NUMI_CONFIG", "configs/training/sft.yaml")
DEVICE = os.getenv("NUMI_DEVICE", "auto")
HISTORY_TOKEN_BUDGET = int(os.getenv("NUMI_HISTORY_TOKENS", "1500"))

app = FastAPI(
    title="Numi-OT.01 API",
    version="0.1.0",
    description="OpenAI-compatible chat API for Numi",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("NUMI_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)


@lru_cache(maxsize=1)
def get_engine() -> NumiEngine:
    return NumiEngine.from_checkpoint(
        checkpoint=CHECKPOINT,
        config_path=CONFIG_PATH,
        device=DEVICE,
    )


@app.on_event("startup")
def warmup() -> None:
    if os.getenv("NUMI_WARMUP", "1") == "1":
        try:
            get_engine()
        except Exception as exc:
            logging.getLogger("numi.api").warning("warmup failed: %s", exc)


@app.get("/health")
def health() -> dict[str, object]:
    try:
        engine = get_engine()
        return {
            "status": "ok",
            "model": MODEL_ID,
            "loaded": True,
            "device": str(engine.device),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "model": MODEL_ID,
            "loaded": False,
            "error": str(exc)[:200],
        }


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return METRICS.snapshot()


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "numination"}],
    }


def _run_generation(request: ChatCompletionRequest):
    engine = get_engine()
    requested = request.model or MODEL_ID
    if requested not in {MODEL_ID, "numi", "numi-ot-0.1", "Numi-OT.01"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{requested}'. Available: '{MODEL_ID}'.",
        )
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    messages = trim_messages(messages, max_tokens=HISTORY_TOKEN_BUDGET)
    with inference_slot():
        return engine.generate(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(_stream_chat(request), media_type="text/event-stream")

    try:
        generation = _run_generation(request)
        METRICS.record_chat(
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
        )
    except HTTPException:
        METRICS.record_chat(prompt_tokens=0, completion_tokens=0, error=True)
        raise
    except Exception as exc:
        METRICS.record_chat(prompt_tokens=0, completion_tokens=0, error=True)
        raise HTTPException(status_code=502, detail=f"Inference failed: {exc}") from exc

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        model=MODEL_ID,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=generation.text),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.prompt_tokens + generation.completion_tokens,
        ),
    )


def _stream_chat(request: ChatCompletionRequest):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        generation = _run_generation(request)
        METRICS.record_chat(
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
        )
    except Exception as exc:
        METRICS.record_chat(prompt_tokens=0, completion_tokens=0, error=True)
        yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
        yield "data: [DONE]\n\n"
        return

    pieces = generation.text.split(" ") if generation.text else [""]
    for i, piece in enumerate(pieces):
        token = piece if i == 0 else " " + piece
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat")
def chat_simple(request: ChatCompletionRequest) -> dict[str, object]:
    if request.stream:
        raise HTTPException(status_code=400, detail="Use /v1/chat/completions for stream=true")
    response = chat_completions(request)
    return {"model": response.model, "response": response.choices[0].message.content}
