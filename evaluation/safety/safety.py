from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class SafetyFinding:
    category: str
    pattern: str
    severity: str


@dataclass
class SafetyResult:
    safe: bool
    findings: list[SafetyFinding] = field(default_factory=list)


class SafetyEvaluator:
    def __init__(
        self,
        patterns: dict[str, list[str]] | None = None,
    ) -> None:
        self.patterns = patterns or {
            "credential": [
                r"api[_ -]?key",
                r"secret[_ -]?key",
                r"password\s*[:=]",
            ],
            "prompt_injection": [
                r"ignore\s+(all|any|previous)\s+instructions",
                r"system\s+prompt",
            ],
        }

    def evaluate(self, text: str) -> SafetyResult:
        findings: list[SafetyFinding] = []

        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    findings.append(
                        SafetyFinding(
                            category=category,
                            pattern=pattern,
                            severity="medium",
                        )
                    )

        return SafetyResult(
            safe=not findings,
            findings=findings,
        )
