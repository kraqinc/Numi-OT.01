"""API key authentication for the Numi inference server."""

from __future__ import annotations

import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_FORBIDDEN_KEYS = {
    "",
    "change-me",
    "change-me-long-secret",
    "your-api-key",
    "secret",
    "password",
    "numi",
    "test",
    "1234567890123456",
}


def expected_api_key() -> str | None:
    key = os.getenv("NUMI_API_KEY") or os.getenv("API_KEY")
    if not key:
        return None
    key = key.strip()
    if key.lower() in _FORBIDDEN_KEYS or len(key) < 24:
        return None
    return key


def require_api_key_configured() -> str:
    env = os.getenv("NUMI_ENV", os.getenv("ENV", "development")).lower()
    key = expected_api_key()
    if key is not None:
        return key
    if env in {"production", "prod", "staging"}:
        raise RuntimeError(
            "NUMI_API_KEY must be set to a strong secret (>=24 chars) "
            "when NUMI_ENV is production/staging. Refusing to start."
        )
    return ""


class APIKeyMiddleware(BaseHTTPMiddleware):
    PUBLIC_PATHS = {"/health", "/live", "/ready", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        key = expected_api_key()
        env = os.getenv("NUMI_ENV", os.getenv("ENV", "development")).lower()

        if key is None:
            if env in {"production", "prod", "staging"}:
                return JSONResponse(
                    {"detail": "API key not configured"}, status_code=503
                )
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        token = ""
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        elif auth.startswith("Api-Key "):
            token = auth[8:].strip()

        header_key = request.headers.get("X-API-Key", "").strip()
        provided = token or header_key
        if not provided or not secrets.compare_digest(provided, key):
            return JSONResponse(
                {"detail": "Invalid or missing API key"}, status_code=401
            )

        return await call_next(request)
