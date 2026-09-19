from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class BenchmarkCase:
    prompt: str
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    name: str
    score: float
    total: int
    correct: int
    details: list[dict[str, Any]] = field(default_factory=list)


class Benchmark:
    def __init__(
        self,
        name: str,
        cases: Iterable[BenchmarkCase],
        evaluator: Callable[[Any, BenchmarkCase], bool] | None = None,
    ) -> None:
        self.name = name
        self.cases = list(cases)
        self.evaluator = evaluator

    def run(
        self,
        generate: Callable[[str], Any],
    ) -> BenchmarkResult:
        details: list[dict[str, Any]] = []
        correct = 0

        for case in self.cases:
            prediction = generate(case.prompt)

            if self.evaluator is not None:
                ok = bool(self.evaluator(prediction, case))
            else:
                ok = prediction == case.expected

            correct += int(ok)
            details.append(
                {
                    "prompt": case.prompt,
                    "expected": case.expected,
                    "prediction": prediction,
                    "correct": ok,
                    "metadata": case.metadata,
                }
            )

        total = len(self.cases)
        score = correct / total if total else 0.0

        return BenchmarkResult(
            name=self.name,
            score=score,
            total=total,
            correct=correct,
            details=details,
        )
