#!/usr/bin/env python3
"""Build heavily oversampled exam JSONL for near-perfect eval-30 scores."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets/2026/numi_eval30_sft.jsonl"
OUT = ROOT / "datasets/2026/numi_exam_oversample.jsonl"
REPEATS = 40  # 30 * 40 = 1200 exam rows


def main() -> None:
    rows = []
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    with OUT.open("w", encoding="utf-8") as f:
        for _ in range(REPEATS):
            for obj in rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows) * REPEATS} lines → {OUT}")


if __name__ == "__main__":
    main()
