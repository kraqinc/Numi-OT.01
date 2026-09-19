from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class DataRecord:
    id: str
    text: str
    source: str
    language: str = "unknown"
    domain: str = "general"
    published_at: str | None = None
    retrieved_at: str | None = None
    quality: float = 0.0
    license: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataRecord":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source=str(data.get("source", "unknown")),
            language=str(data.get("language", "unknown")),
            domain=str(data.get("domain", "general")),
            published_at=data.get("published_at"),
            retrieved_at=data.get(
                "retrieved_at",
                datetime.utcnow().isoformat(),
            ),
            quality=float(data.get("quality", 0.0)),
            license=data.get("license"),
        )
