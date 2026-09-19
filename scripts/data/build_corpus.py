from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import yaml

from data.ingestion.streaming import iter_source
from data.mixing.stream_mix import weighted_mix
from data.processing.corpus import (
    now_iso,
    process_text,
)


def extract_text(
    record: dict[str, Any],
    text_field: str,
) -> str | None:
    value = record.get(text_field)

    if isinstance(value, str):
        return value

    return None


def build(
    config_path: str,
    *,
    max_records: int | None = None,
) -> None:
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    output_config = config["output"]
    processing_config = config["processing"]

    output_dir = Path(
        output_config["directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    enabled_sources = [
        source
        for source in config["sources"]
        if source.get("enabled", False)
        and source.get("mode") == "cpt"
    ]

    streams = {
        source["id"]: iter_source(source)
        for source in enabled_sources
    }

    weights = {
        source["id"]: float(
            source.get("weight", 1.0)
        )
        for source in enabled_sources
    }

    source_by_id = {
        source["id"]: source
        for source in enabled_sources
    }

    seen_hashes: set[str] = set()

    shard_index = 0
    records_in_shard = 0
    total_records = 0
    total_rejected = 0

    shard_handle = None

    def open_shard():
        nonlocal shard_index
        nonlocal records_in_shard

        path = (
            output_dir
            / f"corpus-{shard_index:05d}.jsonl.gz"
        )

        shard_index += 1
        records_in_shard = 0

        return gzip.open(
            path,
            "wt",
            encoding="utf-8",
        )

    try:
        shard_handle = open_shard()

        for source_id, raw_record in weighted_mix(
            streams,
            weights,
        ):
            if (
                max_records is not None
                and total_records >= max_records
            ):
                break

            source = source_by_id[source_id]

            text = extract_text(
                raw_record,
                source["text_field"],
            )

            if text is None:
                total_rejected += 1
                continue

            processed = process_text(
                text,
                min_characters=int(
                    processing_config["min_characters"]
                ),
                max_characters=int(
                    processing_config["max_characters"]
                ),
            )

            if processed is None:
                total_rejected += 1
                continue

            clean_text, digest, quality = processed

            if digest in seen_hashes:
                continue

            seen_hashes.add(digest)

            output_record = {
                "id": digest,
                "text": clean_text,
                "source": source_id,
                "source_type": source["type"],
                "license": source.get("license"),
                "retrieved_at": now_iso(),
                "quality_score": quality,
                "metadata": {
                    "dataset": source.get(
                        "dataset"
                    ),
                    "dataset_config": source.get(
                        "name"
                    ),
                    "split": source.get(
                        "split"
                    ),
                },
            }

            shard_handle.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            total_records += 1
            records_in_shard += 1

            if (
                records_in_shard
                >= output_config["records_per_shard"]
            ):
                shard_handle.close()
                shard_handle = open_shard()

    finally:
        if shard_handle is not None:
            shard_handle.close()

    manifest = {
        "name": config["name"],
        "version": config["version"],
        "target_date": config["target_date"],
        "records": total_records,
        "rejected": total_rejected,
        "deduplication": "sha256",
        "generated_at": now_iso(),
    }

    manifest_path = (
        output_dir / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Numi CPT corpus."
    )

    parser.add_argument(
        "--config",
        default="configs/data/corpus.yaml",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    build(
        args.config,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()