from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset


def iter_jsonl(
    path: str | Path,
) -> Iterator[dict[str, Any]]:
    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"JSONL source does not exist: {source}"
        )

    with source.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {source}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected an object on line {line_number} "
                    f"of {source}"
                )

            yield record


def iter_huggingface(
    *,
    dataset: str,
    name: str | None,
    split: str,
    revision: str | None = None,
) -> Iterator[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "path": dataset,
        "split": split,
        "streaming": True,
    }

    if name:
        kwargs["name"] = name

    if revision:
        kwargs["revision"] = revision

    streamed = load_dataset(**kwargs)

    for record in streamed:
        if not isinstance(record, dict):
            record = dict(record)

        yield record


def iter_source(
    source: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    source_type = source["type"]

    if source_type == "jsonl":
        yield from iter_jsonl(source["path"])
        return

    if source_type == "huggingface":
        yield from iter_huggingface(
            dataset=source["dataset"],
            name=source.get("name"),
            split=source.get("split", "train"),
            revision=source.get("revision"),
        )
        return

    raise ValueError(
        f"Unsupported source type: {source_type}"
    )