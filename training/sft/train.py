from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from tokenizer.templates.chat import NumiChatTemplate


@dataclass
class SFTExample:
    text: str


class ConversationDataset(Dataset):
    """Loads JSONL conversations and renders them with the Numi chat template."""

    def __init__(self, path: str | Path, max_length: int = 512) -> None:
        self.path = Path(path)
        self.max_length = max_length
        self.tokens = NumiSpecialTokens()
        self.template = NumiChatTemplate.from_special_tokens(self.tokens)
        self.examples: list[SFTExample] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record.get("messages", [])
                text = self.template.render(messages, add_generation_prompt=False)
                self.examples.append(SFTExample(text=text))

        if not self.examples:
            raise ValueError(f"No examples found in {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTExample:
        return self.examples[index]


class SimpleTokenizer:
    """
    Minimal deterministic tokenizer for smoke SFT runs.

    Maps UTF-8 bytes + special tokens into a fixed vocab.
    Replace with a real trained Numi tokenizer when available.
    """

    def __init__(self, vocab_size: int = 128256) -> None:
        self.special = NumiSpecialTokens()
        self.special_list = list(self.special.all())
        self.vocab_size = max(vocab_size, 256 + len(self.special_list) + 8)
        self.pad_id = 0
        self.unk_id = 1
        self.special_to_id = {
            tok: 2 + i for i, tok in enumerate(self.special_list)
        }
        self.byte_offset = 2 + len(self.special_list)

    def encode(self, text: str, max_length: int) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in self.special_to_id.items():
                if text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            b = text[i].encode("utf-8")
            for byte in b:
                ids.append(self.byte_offset + (byte % 256))
            i += 1
            if len(ids) >= max_length:
                break
        return ids[:max_length]

    def pad_batch(
        self, sequences: list[list[int]], max_length: int
    ) -> tuple[Tensor, Tensor]:
        batch = []
        labels = []
        for seq in sequences:
            padded = seq + [self.pad_id] * (max_length - len(seq))
            label = seq + [-100] * (max_length - len(seq))
            batch.append(padded[:max_length])
            labels.append(label[:max_length])
        return (
            torch.tensor(batch, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_fn(
    examples: list[SFTExample],
    tokenizer: SimpleTokenizer,
    max_length: int,
) -> dict[str, Tensor]:
    sequences = [tokenizer.encode(ex.text, max_length=max_length) for ex in examples]
    # Ensure at least length 2 for causal LM shift
    sequences = [seq if len(seq) >= 2 else seq + [tokenizer.pad_id] for seq in sequences]
    max_len = min(max_length, max(len(s) for s in sequences))
    input_ids, labels = tokenizer.pad_batch(sequences, max_len)
    return {"input_ids": input_ids, "labels": labels}


def get_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: dict[str, Any]) -> NumiForCausalLM:
    model_cfg = NumiConfig(**cfg["model"])
    return NumiForCausalLM(model_cfg)


def train(config: dict[str, Any]) -> Path:
    random.seed(config["training"]["seed"])
    torch.manual_seed(config["training"]["seed"])

    device = get_device(config["training"].get("device", "auto"))
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ConversationDataset(
        config["data"]["path"],
        max_length=config["data"]["max_seq_length"],
    )
    tokenizer = SimpleTokenizer(vocab_size=config["model"]["vocab_size"])

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer, config["data"]["max_seq_length"]
        ),
    )

    model = build_model(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = (
        math.ceil(len(loader) / config["training"]["gradient_accumulation_steps"])
        * config["training"]["num_epochs"]
    )
    warmup_steps = max(1, int(total_steps * config["training"]["warmup_ratio"]))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    print(f"Device: {device}")
    print(f"Examples: {len(dataset)}")
    print(f"Steps: {total_steps}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config["training"]["num_epochs"]):
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            assert loss is not None

            loss = loss / config["training"]["gradient_accumulation_steps"]
            loss.backward()

            if (batch_idx + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["max_grad_norm"]
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = config["training"]["learning_rate"] * lr_at(global_step)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % config["training"]["logging_steps"] == 0:
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={loss.item() * config['training']['gradient_accumulation_steps']:.4f}"
                    )

                if global_step % config["training"]["save_steps"] == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": config,
                            "step": global_step,
                        },
                        ckpt,
                    )
                    print(f"Saved {ckpt}")

    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "config": config, "step": global_step}, final_path)
    print(f"Training complete → {final_path}")
    return final_path
