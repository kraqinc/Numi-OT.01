#!/usr/bin/env python3
"""Teach Numi a new fact without re-training weights.

Examples:
  python scripts/learn_fact.py "Our Pro plan is $9/mo via PayPal."
  python scripts/learn_fact.py --file ./release_notes.md
  python scripts/learn_fact.py --title "pricing" "Starter 5000 credits"
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARN = ROOT / "knowledge" / "learn"


def slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")
    return (s[:40] or "fact")


def main() -> None:
    p = argparse.ArgumentParser(description="Add knowledge for Numi RAG")
    p.add_argument("text", nargs="?", help="Fact or paragraph to learn")
    p.add_argument("--file", type=Path, help="Copy a markdown/text file into learn/")
    p.add_argument("--title", default="", help="Filename hint")
    args = p.parse_args()

    LEARN.mkdir(parents=True, exist_ok=True)

    if args.file:
        src = args.file.expanduser().resolve()
        if not src.exists():
            raise SystemExit(f"File not found: {src}")
        dest = LEARN / src.name
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Learned file → {dest}")
        return

    if not args.text or not args.text.strip():
        raise SystemExit("Provide text or --file")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = slug(args.title or args.text.split(".")[0])
    dest = LEARN / f"{ts}_{name}.md"
    body = (
        f"# Learned {ts}\n\n"
        f"{args.text.strip()}\n"
    )
    dest.write_text(body, encoding="utf-8")
    print(f"Learned fact → {dest}")
    print("Next chat with RAG on will see this (NUMI_RAG=1).")


if __name__ == "__main__":
    main()
