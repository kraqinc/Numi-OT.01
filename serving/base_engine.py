"""Inference engine for PEFT LoRA adapters on a HF causal LM base."""

from __future__ import annotations

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

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
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
    ) -> GenerationResult:
        messages = ensure_system_message(messages, force_server_system=True)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerationResult(
            text=text.strip(),
            prompt_tokens=prompt_len,
            completion_tokens=int(new_tokens.numel()),
        )
