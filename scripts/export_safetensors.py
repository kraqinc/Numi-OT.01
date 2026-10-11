#!/usr/bin/env python3
"""Export a Numi .pt checkpoint to safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to .safetensors")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit("pip install safetensors") from exc

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    # tensors only
    tensors = {k: v.contiguous() for k, v in state.items() if torch.is_tensor(v)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    print(f"Wrote {out} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
