from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass
class BrowserRequest:
    url: str
    timeout: float = 10.0
    max_bytes: int = 1_000_000


@dataclass
class BrowserResult:
    url: str
    status: int
    content_type: str
    body: bytes


class BrowserTool:
    def fetch(self, request: BrowserRequest) -> BrowserResult:
        parsed = urlparse(request.url)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only HTTP and HTTPS URLs are supported.")

        http_request = Request(
            request.url,
            headers={"User-Agent": "Numi-OT.01/0.1"},
        )

        with urlopen(http_request, timeout=request.timeout) as response:
            body = response.read(request.max_bytes)

            return BrowserResult(
                url=request.url,
                status=response.status,
                content_type=response.headers.get_content_type(),
                body=body,
            )
