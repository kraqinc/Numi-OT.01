from __future__ import annotations

from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    import json

    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["load_json_config"]
