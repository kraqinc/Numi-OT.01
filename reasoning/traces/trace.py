from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ReasoningStep:
    action: str
    observation: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    request: str
    steps: list[ReasoningStep] = field(default_factory=list)
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def add(
        self,
        action: str,
        observation: str = "",
        confidence: float = 0.0,
        **metadata: Any,
    ) -> ReasoningStep:
        step = ReasoningStep(
            action=action,
            observation=observation,
            confidence=max(0.0, min(1.0, confidence)),
            metadata=metadata,
        )
        self.steps.append(step)
        return step

    @property
    def final_confidence(self) -> float:
        if not self.steps:
            return 0.0
        return self.steps[-1].confidence
