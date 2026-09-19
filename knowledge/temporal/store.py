from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TemporalDocument:
    id: str
    text: str
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def is_valid_at(self, timestamp: datetime | None = None) -> bool:
        timestamp = timestamp or datetime.now(timezone.utc)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        if self.valid_from is not None and timestamp < self.valid_from:
            return False

        if self.valid_until is not None and timestamp >= self.valid_until:
            return False

        return True


class TemporalStore:
    def __init__(self) -> None:
        self._documents: dict[str, TemporalDocument] = {}

    def upsert(self, document: TemporalDocument) -> None:
        self._documents[document.id] = document

    def get(
        self,
        document_id: str,
        timestamp: datetime | None = None,
    ) -> TemporalDocument | None:
        document = self._documents.get(document_id)

        if document is None:
            return None

        return document if document.is_valid_at(timestamp) else None

    def active(
        self,
        timestamp: datetime | None = None,
    ) -> list[TemporalDocument]:
        return [
            document
            for document in self._documents.values()
            if document.is_valid_at(timestamp)
        ]
