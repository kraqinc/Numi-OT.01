#!/usr/bin/env python3
"""Continued pretraining with streaming text batches."""

from __future__ import annotations

import argparse

import yaml

from training.cpt.train import train_cpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Numi continued pretraining (CPT)")
    parser.add_argument("--config", default="configs/training/cpt.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.data:
        config["data"]["path"] = args.data
    if args.device:
        config["training"]["device"] = args.device
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps

    train_cpt(config)


if __name__ == "__main__":
    main()
