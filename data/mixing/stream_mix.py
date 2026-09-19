from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def weighted_mix(
    streams: dict[str, Iterator[dict[str, Any]]],
    weights: dict[str, float],
    *,
    slots_per_cycle: int = 1000,
) -> Iterator[tuple[str, dict[str, Any]]]:
    active = {
        name: iterator
        for name, iterator in streams.items()
        if name in weights and weights[name] > 0
    }

    if not active:
        return

    quotas: dict[str, int] = {}

    for name in active:
        quotas[name] = max(
            1,
            round(
                weights[name]
                * slots_per_cycle
            ),
        )

    schedule: list[str] = []

    for name, quota in quotas.items():
        schedule.extend(
            [name] * quota
        )

    while active:
        exhausted: list[str] = []

        for name in schedule:
            if name not in active:
                continue

            try:
                record = next(
                    active[name]
                )
            except StopIteration:
                exhausted.append(name)
                continue

            yield name, record

        for name in exhausted:
            active.pop(name, None)