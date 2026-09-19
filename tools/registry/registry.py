from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def call(self, **kwargs: Any) -> Any:
        if not self.enabled:
            raise RuntimeError(f"Tool '{self.name}' is disabled.")
        return self.handler(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not tool.name:
            raise ValueError("Tool name cannot be empty.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)

        if tool is None:
            raise KeyError(f"Unknown tool: {name}")

        return tool.call(**kwargs)
