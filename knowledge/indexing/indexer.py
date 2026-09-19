from __future__ import annotations

from dataclasses import dataclass, field
import math
import re


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


class InMemoryIndex:
    """Small reference index used for development and tests."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self._terms: dict[str, set[str]] = {}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def add(self, document: Document) -> None:
        self.documents[document.id] = document
        self._terms[document.id] = self._tokenize(document.text)

    def add_many(self, documents: list[Document]) -> None:
        for document in documents:
            self.add(document)

    def remove(self, document_id: str) -> None:
        self.documents.pop(document_id, None)
        self._terms.pop(document_id, None)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        if top_k <= 0:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[Document, float]] = []

        for document_id, terms in self._terms.items():
            intersection = len(query_terms & terms)
            if not intersection:
                continue

            score = intersection / math.sqrt(len(query_terms) * len(terms))
            scored.append((self.documents[document_id], score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
