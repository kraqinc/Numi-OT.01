from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_checkpoint_path(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint}"
        )

    if not checkpoint.is_dir():
        raise ValueError(
            f"Checkpoint path must be a directory: {checkpoint}"
        )

    return checkpoint


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = validate_checkpoint_path(path)

    return {
        "path": str(checkpoint),
        "config": str(checkpoint / "config.json"),
        "weights": [
            str(file)
            for file in sorted(checkpoint.glob("*.safetensors"))
        ],
    }
