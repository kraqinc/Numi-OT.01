#!/usr/bin/env python3
"""Minimal DPO training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.alignment.dpo import train_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Numi with DPO")
    parser.add_argument("--config", default="configs/training/dpo.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.data:
        config["data"]["path"] = args.data
    if args.sft_checkpoint:
        config["training"]["sft_checkpoint"] = args.sft_checkpoint
    if args.device:
        config["training"]["device"] = args.device

    train_dpo(config)


if __name__ == "__main__":
    main()
