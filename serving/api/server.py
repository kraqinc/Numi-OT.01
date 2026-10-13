from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from serving.concurrency import inference_slot
from serving.history import trim_messages
from serving.metrics import METRICS
from serving.middleware.auth import APIKeyMiddleware, require_api_key_configured
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
logger = logging.getLogger("numi.api")

MODEL_ID = os.getenv("NUMI_MODEL_ID", "numi-ot-0.1")
# peft = own SmolLM2+LoRA (independent). cloud = external OpenAI-compatible.
# local = toy Numi MoE .pt
BACKEND = os.getenv("NUMI_BACKEND", "peft").lower()
CHECKPOINT = os.getenv("NUMI_CHECKPOINT")
CONFIG_PATH = os.getenv("NUMI_CONFIG", "configs/training/sft.yaml")
DEVICE = os.getenv("NUMI_DEVICE", "auto")
HISTORY_TOKEN_BUDGET = int(os.getenv("NUMI_HISTORY_TOKENS", "6000"))
PEFT_BASE = os.getenv("NUMI_PEFT_BASE", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
PEFT_ADAPTER = os.getenv("NUMI_PEFT_ADAPTER", "outputs/lora_frontier/final")
PEFT_4BIT = os.getenv("NUMI_PEFT_4BIT", "0") == "1"

_engine: Any = None
_engine_lock = threading.Lock()
_engine_error: str | None = None

app = FastAPI(
    title="Numi-OT.01 API",
    version="0.3.0",
    description="OpenAI-compatible chat API for Numination (own model)",
)

_cors = os.getenv("NUMI_CORS_ORIGINS", "*")
_origins = [o.strip() for o in _cors.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-Id"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)


def get_engine() -> Any:
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            if BACKEND == "peft":
                from serving.base_engine import BaseLoRAEngine

                logger.info(
                    "loading peft base=%s adapter=%s 4bit=%s",
                    PEFT_BASE,
                    PEFT_ADAPTER,
                    PEFT_4BIT,
                )
                _engine = BaseLoRAEngine.from_adapter(
                    base_model_id=PEFT_BASE,
                    adapter_path=PEFT_ADAPTER,
                    load_in_4bit=PEFT_4BIT,
                )
            elif BACKEND == "cloud":
                from serving.cloud_engine import CloudEngine

                _engine = CloudEngine.from_env()
                logger.info("backend=cloud model=%s", getattr(_engine, "model", "?"))
            else:
                from serving.engine import NumiEngine

                _engine = NumiEngine.from_checkpoint(
                    checkpoint=CHECKPOINT,
                    config_path=CONFIG_PATH,
                    device=DEVICE,
                )
                logger.info("backend=local device=%s", _engine.device)
            _engine_error = None
            return _engine
        except Exception as exc:
            _engine_error = type(exc).__name__
            logger.exception("engine load failed")
            raise


@app.on_event("startup")
def on_startup() -> None:
    require_api_key_configured()
    logger.info("startup backend=%s model_id=%s", BACKEND, MODEL_ID)
    if os.getenv("NUMI_WARMUP", "1") == "1":
        try:
            get_engine()
            logger.info("warmup ok backend=%s", BACKEND)
        except Exception as exc:
            logger.warning("warmup failed: %s", type(exc).__name__)


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, object]:
    """Readiness: engine must be loaded."""
    if _engine is not None:
        return {"status": "ready", "model": MODEL_ID, "backend": BACKEND}
    return {
        "status": "not_ready",
        "model": MODEL_ID,
        "backend": BACKEND,
        "error": _engine_error or "not_loaded",
    }


@app.get("/health")
def health() -> dict[str, object]:
    if _engine is not None:
        return {
            "status": "ok",
            "model": MODEL_ID,
            "backend": BACKEND,
            "loaded": True,
            "device": str(getattr(_engine, "device", BACKEND)),
        }
    return {
        "status": "starting" if _engine_error is None else "degraded",
        "model": MODEL_ID,
        "backend": BACKEND,
        "loaded": False,
        "error": _engine_error,
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
    requested = request.model or MODEL_ID
    if requested not in {MODEL_ID, "numi", "numi-ot-0.1", "Numi-OT.01", "numi-ot"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{requested}'. Available: '{MODEL_ID}'.",
        )
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    messages = trim_messages(messages, max_tokens=HISTORY_TOKEN_BUDGET)
    with inference_slot():
        engine = get_engine()
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
        logger.exception("inference failed")
        raise HTTPException(status_code=502, detail="Inference failed") from exc

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
    except Exception:
        METRICS.record_chat(prompt_tokens=0, completion_tokens=0, error=True)
        yield f"data: {json.dumps({'error': {'message': 'Inference failed'}})}\n\n"
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
