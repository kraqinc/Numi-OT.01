from __future__ import annotations


def health_payload(model_id: str) -> dict[str, str]:
    return {"status": "ok", "model": model_id}
