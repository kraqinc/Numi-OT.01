from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM
from tokenizer.factory import load_tokenizer
from training.sft.train import get_device


def stream_text_lines(path: Path) -> Iterator[str]:
    """Stream plain text or JSONL with a 'text' field."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "text" in row:
                    yield str(row["text"])
                elif "messages" in row:
                    # join message contents as crude CPT text
                    yield "\n".join(
                        str(m.get("content", "")) for m in row["messages"]
                    )
            else:
                yield line


def iter_token_batches(
    path: Path,
    tokenizer: Any,
    seq_length: int,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    """Pack streamed tokens into fixed-length batches [B, S]."""
    buffer: list[int] = []
    batch: list[list[int]] = []

    for text in stream_text_lines(path):
        ids = tokenizer.encode(text, max_length=None)
        buffer.extend(ids)
        while len(buffer) >= seq_length:
            chunk = buffer[:seq_length]
            buffer = buffer[seq_length:]
            batch.append(chunk)
            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []

    if batch:
        # pad last incomplete batch
        pad_id = getattr(tokenizer, "pad_id", 0)
        while len(batch) < batch_size:
            batch.append([pad_id] * seq_length)
        yield torch.tensor(batch, dtype=torch.long)


def train_cpt(config: dict[str, Any]) -> Path:
    device = get_device(config["training"].get("device", "auto"))
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        config["data"].get("tokenizer_path"),
        vocab_size=config["model"]["vocab_size"],
    )
    if tokenizer.vocab_size != config["model"]["vocab_size"]:
        config["model"]["vocab_size"] = tokenizer.vocab_size

    model = NumiForCausalLM(NumiConfig(**config["model"])).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    seq_length = int(config["data"]["seq_length"])
    batch_size = int(config["training"]["batch_size"])
    max_steps = int(config["training"]["max_steps"])
    accum = int(config["training"].get("gradient_accumulation_steps", 1))
    data_path = Path(config["data"]["path"])

    print(f"Device: {device}")
    print(f"Tokenizer: {type(tokenizer).__name__}")
    print(f"Streaming CPT from {data_path}")

    step = 0
    optimizer.zero_grad(set_to_none=True)

    # cycle the stream until max_steps
    while step < max_steps:
        for batch in iter_token_batches(data_path, tokenizer, seq_length, batch_size):
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            assert loss is not None
            (loss / accum).backward()

            if (step + 1) % accum == 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"].get("max_grad_norm", 1.0)
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % int(config["training"].get("logging_steps", 10)) == 0:
                print(f"step={step} loss={loss.item():.4f}")

            if step % int(config["training"].get("save_steps", 100)) == 0:
                ckpt = output_dir / f"cpt-{step}.pt"
                torch.save({"model": model.state_dict(), "config": config, "step": step}, ckpt)
                print(f"Saved {ckpt}")

            if step >= max_steps:
                break

    final_path = output_dir / "cpt_final.pt"
    torch.save({"model": model.state_dict(), "config": config, "step": step}, final_path)
    print(f"CPT complete → {final_path}")
    return final_path
