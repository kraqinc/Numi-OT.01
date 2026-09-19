from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CorpusRecord:
    document_id: str
    text: str

    source: str
    source_type: str

    url: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None

    language: str | None = None
    license: str | None = None

    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)