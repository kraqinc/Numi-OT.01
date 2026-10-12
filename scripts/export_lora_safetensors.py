#!/usr/bin/env python3
"""Export PEFT LoRA adapter weights to a single .safetensors + meta json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="outputs/lora_frontier/final",
        help="PEFT adapter directory",
    )
    parser.add_argument(
        "--output",
        default="outputs/lora_frontier/numi_lora.safetensors",
        help="Output .safetensors path",
    )
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
        import torch
    except ImportError as exc:
        raise SystemExit("pip install safetensors torch") from exc

    adapter = Path(args.adapter)
    if not adapter.is_absolute():
        adapter = _ROOT / adapter
    if not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}")

    # PEFT saves adapter_model.safetensors or .bin
    st = adapter / "adapter_model.safetensors"
    bin_path = adapter / "adapter_model.bin"

    out = Path(args.output)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    if st.exists():
        # Already safetensors — copy / re-save cleanly
        from safetensors.torch import load_file

        tensors = load_file(str(st))
        save_file(tensors, str(out))
        print(f"Copied adapter safetensors → {out} ({len(tensors)} tensors)")
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        tensors = {k: v.contiguous() for k, v in state.items() if torch.is_tensor(v)}
        save_file(tensors, str(out))
        print(f"Converted bin → {out} ({len(tensors)} tensors)")
    else:
        raise SystemExit(f"No adapter_model.safetensors or .bin in {adapter}")

    meta = {
        "adapter_dir": str(adapter),
        "output": str(out),
        "files": sorted(p.name for p in adapter.iterdir()),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
