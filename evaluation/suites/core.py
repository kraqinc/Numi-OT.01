from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class EvaluationCase:
    name: str
    input: str
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    name: str
    passed: bool
    prediction: Any
    expected: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationSuite:
    def __init__(
        self,
        name: str,
        cases: list[EvaluationCase],
    ) -> None:
        self.name = name
        self.cases = cases

    def evaluate(
        self,
        fn: Callable[[str], Any],
        comparator: Callable[[Any, Any], bool] | None = None,
    ) -> list[EvaluationResult]:
        results = []

        for case in self.cases:
            prediction = fn(case.input)

            if comparator is None:
                passed = prediction == case.expected
            else:
                passed = comparator(prediction, case.expected)

            results.append(
                EvaluationResult(
                    name=case.name,
                    passed=passed,
                    prediction=prediction,
                    expected=case.expected,
                    metadata=case.metadata,
                )
            )

        return results
