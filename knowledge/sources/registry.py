from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class KnowledgeSource:
    name: str
    uri: str
    kind: str = "document"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    loader: Callable[[], list[dict[str, Any]]] | None = None


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, KnowledgeSource] = {}

    def register(self, source: KnowledgeSource) -> None:
        if not source.name:
            raise ValueError("source.name cannot be empty.")
        self._sources[source.name] = source

    def get(self, name: str) -> KnowledgeSource | None:
        return self._sources.get(name)

    def remove(self, name: str) -> None:
        self._sources.pop(name, None)

    def list(self, enabled_only: bool = False) -> list[KnowledgeSource]:
        sources = list(self._sources.values())
        if enabled_only:
            sources = [source for source in sources if source.enabled]
        return sorted(sources, key=lambda source: source.name)

    def load(self, name: str) -> list[dict[str, Any]]:
        source = self.get(name)

        if source is None:
            raise KeyError(f"Unknown knowledge source: {name}")

        if not source.enabled:
            return []

        if source.loader is None:
            return []

        return source.loader()
