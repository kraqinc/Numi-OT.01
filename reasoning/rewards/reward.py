from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


@dataclass
class RewardSignal:
    name: str
    value: float
    weight: float = 1.0

    def weighted(self) -> float:
        return self.value * self.weight


def aggregate_rewards(signals: Iterable[RewardSignal]) -> float:
    signals = list(signals)

    if not signals:
        return 0.0

    total_weight = sum(signal.weight for signal in signals)

    if total_weight == 0:
        return 0.0

    return sum(signal.weighted() for signal in signals) / total_weight
