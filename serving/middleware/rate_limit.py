"""Simple in-memory rate limiter (per client IP)."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int | None = None):
        super().__init__(app)
        self.limit = requests_per_minute or int(os.getenv("NUMI_RATE_LIMIT_RPM", "60"))
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in {"/health", "/metrics"}:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = self.hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        window.append(now)
        return await call_next(request)
