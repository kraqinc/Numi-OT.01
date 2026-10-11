#!/usr/bin/env python3
"""Run smoke evaluation for Numi."""

from __future__ import annotations

import argparse
import json

from evaluation.suites.smoke import run_smoke_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Numi-OT.01")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default="datasets/2026/numi_conversation_seed.jsonl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = run_smoke_eval(
        data_path=args.data,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(json.dumps({
        "perplexity": report.perplexity,
        "identity_ok": report.identity_ok,
        "samples": report.samples,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
