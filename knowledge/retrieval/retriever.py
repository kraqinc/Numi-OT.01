from __future__ import annotations

from dataclasses import dataclass

from knowledge.indexing.indexer import Document, InMemoryIndex


@dataclass
class RetrievalResult:
    document: Document
    score: float


class KnowledgeRetriever:
    def __init__(self, index: InMemoryIndex | None = None) -> None:
        self.index = index or InMemoryIndex()

    def add(self, document: Document) -> None:
        self.index.add(document)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(document=document, score=score)
            for document, score in self.index.search(query, top_k)
        ]

    def context(
        self,
        query: str,
        top_k: int = 5,
        separator: str = "\n\n",
    ) -> str:
        results = self.retrieve(query, top_k)
        return separator.join(result.document.text for result in results)
