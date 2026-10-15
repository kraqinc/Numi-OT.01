"""Offline scoring of eval-30 CSV-style results.

Usage:
  python evaluation/bench_eval30.py path/to/respuestas.csv
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def norm(s: str) -> str:
    s = (s or "").strip()
    s = s.strip("\"'` ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def score_row(expected: str, answer: str) -> bool:
    e, a = norm(expected), norm(answer)
    if e == a:
        return True
    # allow trailing period / same digits only
    if e.rstrip(".") == a.rstrip("."):
        return True
    if re.fullmatch(r"[\d./]+", e) and re.fullmatch(r"[\d./]+", a):
        return e == a
    return e in a and len(a) <= len(e) + 2


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "respuestas.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_model: dict[str, list[bool]] = {}
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        ok = score_row(r.get("expected", ""), r.get("answer", ""))
        m = r.get("model", "?")
        c = r.get("category", "?")
        by_model.setdefault(m, []).append(ok)
        by_cat.setdefault(f"{m}|{c}", []).append(ok)
    print("=== By model ===")
    for m, flags in sorted(by_model.items()):
        n = len(flags)
        print(f"{m}: {sum(flags)}/{n} = {100 * sum(flags) / n:.1f}%")
    print("=== By model|category ===")
    for k, flags in sorted(by_cat.items()):
        n = len(flags)
        print(f"{k}: {sum(flags)}/{n} = {100 * sum(flags) / n:.1f}%")


if __name__ == "__main__":
    main()
