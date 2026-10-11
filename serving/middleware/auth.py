"""Optional shared-secret API key for the Numi inference server."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


def expected_api_key() -> str | None:
    key = os.getenv("NUMI_API_KEY") or os.getenv("API_KEY")
    return key.strip() if key else None


class APIKeyMiddleware(BaseHTTPMiddleware):
    """If NUMI_API_KEY is set, require Authorization: Bearer <key> (except health)."""

    PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        key = expected_api_key()
        if key is None or request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        token = ""
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        elif auth.startswith("Api-Key "):
            token = auth[8:].strip()

        header_key = request.headers.get("X-API-Key", "").strip()
        if token != key and header_key != key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
