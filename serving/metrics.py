"""Process-local metrics counters for the Numi API."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.time)
    requests_total: int = 0
    chat_total: int = 0
    chat_errors: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_chat(self, *, prompt_tokens: int, completion_tokens: int, error: bool = False) -> None:
        with self._lock:
            self.requests_total += 1
            self.chat_total += 1
            if error:
                self.chat_errors += 1
            else:
                self.tokens_prompt += prompt_tokens
                self.tokens_completion += completion_tokens

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "uptime_sec": round(time.time() - self.started_at, 1),
                "requests_total": self.requests_total,
                "chat_total": self.chat_total,
                "chat_errors": self.chat_errors,
                "tokens_prompt": self.tokens_prompt,
                "tokens_completion": self.tokens_completion,
            }


METRICS = Metrics()
