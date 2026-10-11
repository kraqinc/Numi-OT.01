#!/usr/bin/env python3
"""Minimal SFT entrypoint for Numi-OT.01."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.sft.train import train


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Numi with supervised fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/sft.yaml",
        help="Path to SFT config YAML",
    )
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    config = load_config(args.config)

    if args.data:
        config["data"]["path"] = args.data
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device

    train(config)


if __name__ == "__main__":
    main()
