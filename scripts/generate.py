#!/usr/bin/env python3
"""Generate text with a Numi checkpoint (or a fresh randomly-initialized model)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM
from tokenizer.factory import load_tokenizer
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from tokenizer.templates.chat import NumiChatTemplate


def load_model(
    checkpoint: str | None,
    config_path: str | None,
    device: torch.device,
) -> tuple[NumiForCausalLM, dict, object]:
    if config_path is None:
        config_path = "configs/training/sft.yaml"
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
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model"], strict=False)
        except Exception:
            pass

    model.to(device)
    model.eval()
    return model, config, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with Numi-OT.01")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/training/sft.yaml")
    parser.add_argument("--prompt", type=str, default="Hola")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config, tokenizer = load_model(args.checkpoint, args.config, device)
    special = NumiSpecialTokens()
    template = NumiChatTemplate.from_special_tokens(special)

    messages = [
        {"role": "system", "content": "You are Numi, the AI assistant of Numination."},
        {"role": "user", "content": args.prompt},
    ]
    text = template.render(messages, add_generation_prompt=True)
    input_ids = torch.tensor(
        [tokenizer.encode(text, max_length=config["data"].get("max_seq_length", 512))],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    new_ids = output_ids[0, input_ids.shape[1] :].tolist()
    print(tokenizer.decode(new_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
