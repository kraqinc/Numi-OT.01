"""API key authentication for the Numi inference server."""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Known placeholder values that must never be accepted as real secrets
_FORBIDDEN_KEYS = {
    "",
    "change-me",
    "change-me-long-secret",
    "your-api-key",
    "secret",
    "password",
    "numi",
    "test",
}


def expected_api_key() -> str | None:
    key = os.getenv("NUMI_API_KEY") or os.getenv("API_KEY")
    if not key:
        return None
    key = key.strip()
    if key.lower() in _FORBIDDEN_KEYS or len(key) < 16:
        return None
    return key


def require_api_key_configured() -> str:
    """Fail closed in production if no strong key is set."""
    env = os.getenv("NUMI_ENV", os.getenv("ENV", "development")).lower()
    key = expected_api_key()
    if key is not None:
        return key
    if env in {"production", "prod", "staging"}:
        raise RuntimeError(
            "NUMI_API_KEY must be set to a strong secret (>=16 chars) "
            "when NUMI_ENV is production/staging. Refusing to start."
        )
    return ""


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require Authorization: Bearer <key> except on public paths.

    In production/staging a key is mandatory (checked at startup).
    In development, if no key is set, auth is skipped with a warning path.
    """

    PUBLIC_PATHS = {"/health", "/live", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        key = expected_api_key()
        env = os.getenv("NUMI_ENV", os.getenv("ENV", "development")).lower()

        if key is None:
            if env in {"production", "prod", "staging"}:
                raise HTTPException(status_code=503, detail="API key not configured")
            # development only: allow without key
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
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
