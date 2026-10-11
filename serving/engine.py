from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM
from tokenizer.factory import load_tokenizer
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from tokenizer.templates.chat import NumiChatTemplate
from tokenizer.templates.system import ensure_system_message


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class NumiEngine:
    """Lightweight inference engine for the Numi FastAPI server."""

    def __init__(
        self,
        model: NumiForCausalLM,
        tokenizer: Any,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.special = NumiSpecialTokens()
        self.template = NumiChatTemplate.from_special_tokens(self.special)
        self.model_id = config.get("name", "numi-ot-0.1")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | None = None,
        config_path: str = "configs/training/sft.yaml",
        device: str = "auto",
    ) -> NumiEngine:
        if device == "cpu":
            dev = torch.device("cpu")
        elif device == "cuda":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        tokenizer = load_tokenizer(
            config.get("data", {}).get("tokenizer_path"),
            vocab_size=config["model"]["vocab_size"],
        )
        if tokenizer.vocab_size != config["model"]["vocab_size"]:
            config["model"]["vocab_size"] = tokenizer.vocab_size

        model = NumiForCausalLM(NumiConfig(**config["model"]))

        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
            try:
                model.load_state_dict(ckpt["model"], strict=False)
            except Exception:
                pass
            if "config" in ckpt:
                config = {**config, **ckpt.get("config", {})}

        model.to(dev)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, config=config, device=dev)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = 50,
        inject_system: bool = True,
    ) -> GenerationResult:
        if inject_system:
            messages = ensure_system_message(messages, mode="runtime")

        prompt = self.template.render(messages, add_generation_prompt=True)
        max_length = int(self.config.get("data", {}).get("max_seq_length", 512))
        prompt_ids = self.tokenizer.encode(prompt, max_length=max_length)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5),
                top_k=top_k,
                top_p=top_p,
            )

        new_ids = output_ids[0, input_ids.shape[1] :].tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(new_ids),
        )
