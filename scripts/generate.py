#!/usr/bin/env python3
"""Generate text with a Numi checkpoint (or a fresh randomly-initialized model)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from tokenizer.templates.chat import NumiChatTemplate
from training.sft.train import SimpleTokenizer


def load_model(
    checkpoint: str | None,
    config_path: str | None,
    device: torch.device,
) -> tuple[NumiForCausalLM, dict]:
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        config = ckpt.get("config")
        if config is None and config_path:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        if config is None:
            raise ValueError("Checkpoint has no config and --config was not provided")
        model = NumiForCausalLM(NumiConfig(**config["model"]))
        model.load_state_dict(ckpt["model"])
    else:
        if config_path is None:
            config_path = "configs/training/sft.yaml"
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model = NumiForCausalLM(NumiConfig(**config["model"]))

    model.to(device)
    model.eval()
    return model, config


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

    model, config = load_model(args.checkpoint, args.config, device)
    tokenizer = SimpleTokenizer(vocab_size=config["model"]["vocab_size"])
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

    # Naive decode: map byte ids back when possible
    ids = output_ids[0].tolist()
    pieces: list[str] = []
    id_to_special = {v: k for k, v in tokenizer.special_to_id.items()}
    for tid in ids:
        if tid in id_to_special:
            pieces.append(id_to_special[tid])
        elif tid >= tokenizer.byte_offset:
            byte = (tid - tokenizer.byte_offset) % 256
            try:
                pieces.append(bytes([byte]).decode("utf-8", errors="ignore"))
            except Exception:
                pieces.append("?")
        else:
            pieces.append("")

    print("".join(pieces))


if __name__ == "__main__":
    main()
