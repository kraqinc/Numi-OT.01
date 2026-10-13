"""Inference engine for PEFT LoRA adapters on a HF causal LM base."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tokenizer.templates.system import ensure_system_message


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


def _clean_completion(text: str) -> str:
    text = text.strip()
    # Drop accidental role echoes
    for marker in ("\nuser:", "\nassistant:", "\nUser:", "\nAssistant:", "<|im_end|>"):
        if marker in text:
            text = text.split(marker)[0]
    # Collapse runaway same-word loops ("que que que")
    text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"([.!?…])\1+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class BaseLoRAEngine:
    def __init__(self, model: Any, tokenizer: Any, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_adapter(
        cls,
        base_model_id: str,
        adapter_path: str,
        *,
        load_in_4bit: bool = True,
        device: str = "auto",
    ) -> BaseLoRAEngine:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        adapter = Path(adapter_path)
        is_local = adapter.exists()

        if is_local:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(adapter), local_files_only=True
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            peft_id = str(adapter)
        else:
            # HF hub id e.g. user/numi-ot-lora
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            peft_id = adapter_path

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": False,
        }
        if load_in_4bit and torch.cuda.is_available():
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        try:
            base = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
        except TypeError:
            # older transformers used dtype= instead of torch_dtype
            load_kwargs.pop("torch_dtype", None)
            load_kwargs["dtype"] = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            base = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
        except ValueError as exc:
            if "quantization" in str(exc).lower():
                load_kwargs.pop("quantization_config", None)
                base = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
            else:
                raise

        model = PeftModel.from_pretrained(base, peft_id)
        model.eval()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model=model, tokenizer=tokenizer, device=dev)

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.18,
        no_repeat_ngram_size: int = 4,
    ) -> GenerationResult:
        messages = ensure_system_message(messages, force_server_system=True)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                + "\nassistant:"
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(self.tokenizer, "model_max_length", 2048) or 2048, 2048),
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        # Cap runaway generation on CPU / small models
        max_tokens = max(16, min(int(max_tokens), 512))

        do_sample = temperature is not None and temperature > 0.05
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(max(0.3, min(temperature, 1.2)))
            gen_kwargs["top_p"] = float(max(0.7, min(top_p, 0.95)))
            gen_kwargs["top_k"] = 50
        else:
            gen_kwargs["do_sample"] = False

        out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = _clean_completion(text)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(new_tokens.numel()),
        )
