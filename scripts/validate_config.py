#!/usr/bin/env python3
"""Validate a training YAML against NumiConfig constraints."""

from __future__ import annotations

import argparse
import sys

import yaml

from model.configs.numi_config import NumiConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model_cfg = data.get("model") or {}
    try:
        cfg = NumiConfig(**model_cfg)
    except Exception as exc:
        print(f"INVALID: {exc}")
        sys.exit(1)

    print("OK")
    print(f"  layers={cfg.num_hidden_layers} hidden={cfg.hidden_size}")
    print(f"  heads={cfg.num_attention_heads} kv={cfg.num_key_value_heads}")
    print(f"  experts={cfg.num_experts} top_k={cfg.num_experts_per_tok}")
    print(f"  vocab={cfg.vocab_size}")


if __name__ == "__main__":
    main()
