from __future__ import annotations

import random
from collections.abc import Iterable


def weighted_mix(
    datasets: dict[str, list[dict]],
    weights: dict[str, float],
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)

    pools = {
        name: list(records)
        for name, records in datasets.items()
        if records
    }

    if not pools:
        return []

    total = sum(max(weights.get(name, 0.0), 0.0) for name in pools)

    if total <= 0:
        return [
            record
            for records in pools.values()
            for record in records
        ]

    result: list[dict] = []

    names = list(pools)

    while any(pools.values()):
        available = [name for name in names if pools[name]]

        scores = [
            max(weights.get(name, 0.0), 0.0)
            for name in available
        ]

        if sum(scores) == 0:
            selected = rng.choice(available)
        else:
            selected = rng.choices(
                available,
                weights=scores,
                k=1,
            )[0]

        index = rng.randrange(len(pools[selected]))
        result.append(pools[selected].pop(index))

    return result


def concat_datasets(
    datasets: Iterable[Iterable[dict]],
) -> list[dict]:
    return [
        record
        for dataset in datasets
        for record in dataset
    ]
