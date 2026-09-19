from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RegressionCase:
    name: str
    prompt: str
    expected: str | None = None


@dataclass
class RegressionReport:
    passed: int
    failed: int
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed


class RegressionSuite:
    def __init__(self, cases: list[RegressionCase]) -> None:
        self.cases = cases

    def run(
        self,
        generate: Callable[[str], str],
        comparator: Callable[[str, str], bool] | None = None,
    ) -> RegressionReport:
        passed = 0
        failed = 0
        failures: list[dict[str, str]] = []

        for case in self.cases:
            actual = generate(case.prompt)

            if case.expected is None:
                ok = bool(actual.strip())
            elif comparator:
                ok = comparator(actual, case.expected)
            else:
                ok = actual.strip() == case.expected.strip()

            if ok:
                passed += 1
            else:
                failed += 1
                failures.append(
                    {
                        "name": case.name,
                        "prompt": case.prompt,
                        "expected": case.expected or "",
                        "actual": actual,
                    }
                )

        return RegressionReport(
            passed=passed,
            failed=failed,
            failures=failures,
        )
