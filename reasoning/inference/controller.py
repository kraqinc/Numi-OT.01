from __future__ import annotations

from dataclasses import dataclass

from reasoning.policies.policy import ReasoningPolicy
from reasoning.traces.trace import ReasoningTrace


@dataclass
class ReasoningDecision:
    continue_reasoning: bool
    verify: bool
    step: int
    confidence: float


class ReasoningController:
    def __init__(
        self,
        policy: ReasoningPolicy | None = None,
    ) -> None:
        self.policy = policy or ReasoningPolicy()

    def decide(
        self,
        trace: ReasoningTrace,
        confidence: float,
        needs_tool: bool = False,
    ) -> ReasoningDecision:
        step = len(trace.steps)

        return ReasoningDecision(
            continue_reasoning=self.policy.should_continue(
                step=step,
                confidence=confidence,
                needs_tool=needs_tool,
            ),
            verify=self.policy.should_verify(confidence),
            step=step,
            confidence=confidence,
        )
