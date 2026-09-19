from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReasoningPolicy:
    max_steps: int = 8
    min_confidence: float = 0.5
    allow_tools: bool = True
    require_verification: bool = False

    def should_continue(
        self,
        step: int,
        confidence: float,
        needs_tool: bool = False,
    ) -> bool:
        if step >= self.max_steps:
            return False

        if needs_tool and not self.allow_tools:
            return False

        return confidence < self.min_confidence or needs_tool

    def should_verify(self, confidence: float) -> bool:
        return self.require_verification or confidence < self.min_confidence
