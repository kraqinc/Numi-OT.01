"""Limit concurrent inference jobs (protects small hosts)."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

from fastapi import HTTPException

_MAX = int(os.getenv("NUMI_MAX_CONCURRENT", "2"))
_sem = threading.Semaphore(_MAX)


@contextmanager
def inference_slot(timeout: float = 0.01):
    acquired = _sem.acquire(timeout=timeout)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy, retry later")
    try:
        yield
    finally:
        _sem.release()
