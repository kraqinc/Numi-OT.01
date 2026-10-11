#!/usr/bin/env python3
"""CLI for Numi supervised fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.sft.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Numi SFT")
    parser.add_argument("--config", default="configs/training/sft.yaml")
    parser.add_argument("--device", default=None, help="cpu | cuda | auto")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint .pt to resume weights (and step if present)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.device:
        config.setdefault("training", {})["device"] = args.device
    if args.resume:
        config.setdefault("training", {})["resume_from"] = args.resume

    path = train(config)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
