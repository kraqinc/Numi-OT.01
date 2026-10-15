#!/usr/bin/env python3
"""Train Numi LoRA from YAML (default: 32B frontier)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.base_lora.train import train_lora


def main() -> None:
    p = argparse.ArgumentParser(description="Numi LoRA SFT")
    p.add_argument(
        "--config",
        default="configs/training/lora_frontier.yaml",
        help="YAML config (default: 32B frontier)",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Override base.model_id (e.g. Qwen/Qwen2.5-32B-Instruct)",
    )
    args = p.parse_args()
    path = Path(args.config)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.model_id:
        cfg.setdefault("base", {})["model_id"] = args.model_id
    print("=" * 60)
    print("NUMI LoRA TRAIN")
    print("config:", path)
    print("base:  ", cfg.get("base", {}).get("model_id"))
    print("out:   ", cfg.get("training", {}).get("output_dir"))
    print("=" * 60)
    out = train_lora(cfg)
    print("Done:", out)


if __name__ == "__main__":
    main()
