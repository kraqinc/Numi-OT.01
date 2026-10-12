#!/usr/bin/env python3
"""Fine-tune a frontier open base with LoRA + Numi identity data."""

from __future__ import annotations

import argparse

from training.base_lora.train import train_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/training/lora_frontier.yaml",
        help="YAML config (default: lora_frontier.yaml)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override base.model_id (HF hub id)",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.model_id:
        cfg.setdefault("base", {})["model_id"] = args.model_id
        # write temp? train_lora accepts dict — call train_lora directly
        from training.base_lora.train import train_lora

        path = train_lora(cfg)
    else:
        path = train_from_yaml(args.config)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
