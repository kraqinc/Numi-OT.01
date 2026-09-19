from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass
class ConversionReport:
    source: str
    destination: str
    fields_copied: list[str]
    warnings: list[str]


_FIELD_MAP = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "intermediate_size": "intermediate_size",
    "max_position_embeddings": "max_position_embeddings",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_norm_eps",
    "tie_word_embeddings": "tie_word_embeddings",
}


def convert_qwen_config(
    source: str | Path,
    destination: str | Path,
) -> ConversionReport:
    source_path = Path(source)
    destination_path = Path(destination)

    with source_path.open("r", encoding="utf-8") as handle:
        source_config: dict[str, Any] = json.load(handle)

    converted: dict[str, Any] = {
        "model_type": "numi",
    }

    copied: list[str] = []
    warnings: list[str] = []

    for source_key, target_key in _FIELD_MAP.items():
        if source_key in source_config:
            converted[target_key] = source_config[source_key]
            copied.append(source_key)

    if "num_experts" in source_config:
        converted["num_experts"] = source_config["num_experts"]

    if "num_experts_per_tok" in source_config:
        converted["num_experts_per_tok"] = source_config["num_experts_per_tok"]

    if "num_experts" not in converted:
        warnings.append("num_experts was not present in the source config.")

    if "num_experts_per_tok" not in converted:
        warnings.append(
            "num_experts_per_tok was not present in the source config."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with destination_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return ConversionReport(
        source=str(source_path),
        destination=str(destination_path),
        fields_copied=copied,
        warnings=warnings,
    )
