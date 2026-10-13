"""Simple in-memory rate limiter (per client IP)."""

from __future__ import annotations

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

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.PUBLIC_SKIP:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = self.hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            return JSONResponse(
                {"detail": "Rate limit exceeded"}, status_code=429
            )
        window.append(now)
        return await call_next(request)
