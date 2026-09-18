from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class QwenModelBundle:
    model: AutoModelForCausalLM
    tokenizer: object
    model_name: str


def load_qwen(
    model_name: str,
    *,
    revision: str = "main",
    trust_remote_code: bool = False,
) -> QwenModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )

    model_kwargs = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    model.eval()

    if model.generation_config.pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.eos_token_id

    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    return QwenModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
    )
