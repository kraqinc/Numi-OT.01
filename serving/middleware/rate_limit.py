"""In-memory rate limiter keyed by API key hash or client IP."""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    PUBLIC_SKIP = {"/health", "/live", "/ready", "/metrics"}

    def __init__(self, app, requests_per_minute: int | None = None):
        super().__init__(app)
        self.limit = requests_per_minute or int(os.getenv("NUMI_RATE_LIMIT_RPM", "60"))
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        token = ""
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        elif auth.startswith("Api-Key "):
            token = auth[8:].strip()
        token = token or request.headers.get("X-API-Key", "").strip()
        if token:
            return "k:" + hashlib.sha256(token.encode()).hexdigest()[:24]
        host = request.client.host if request.client else "unknown"
        return "ip:" + host

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.PUBLIC_SKIP:
            return await call_next(request)

        key = self._client_key(request)
        now = time.time()
        window = self.hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            return JSONResponse(
                {"detail": "Rate limit exceeded", "retry_after": 60},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        window.append(now)
        return await call_next(request)
