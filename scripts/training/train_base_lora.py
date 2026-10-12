#!/usr/bin/env python3
"""Fine-tune a frontier open base with LoRA + Numi identity data."""

from __future__ import annotations

import sys
from pathlib import Path

# Colab / script runs without pip install -e .
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import yaml

from training.base_lora.train import train_from_yaml, train_lora


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

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _ROOT / config_path

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.model_id:
        cfg.setdefault("base", {})["model_id"] = args.model_id

    # Resolve relative data paths from repo root
    paths = cfg.get("data", {}).get("paths", [])
    resolved = []
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = _ROOT / pp
        resolved.append(str(pp))
    cfg.setdefault("data", {})["paths"] = resolved

    out = cfg.get("training", {}).get("output_dir", "outputs/lora_frontier")
    out_p = Path(out)
    if not out_p.is_absolute():
        cfg.setdefault("training", {})["output_dir"] = str(_ROOT / out_p)

    path = train_lora(cfg)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
