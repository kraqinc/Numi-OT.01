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
from tokenizer.factory import SimpleTokenizer, load_tokenizer
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from tokenizer.templates.chat import NumiChatTemplate
from training.checkpointing.checkpoints import save_checkpoint
from training.lora.lora import LoRAConfig, apply_lora, mark_only_lora_as_trainable

__all__ = [
    "ConversationDataset",
    "SimpleTokenizer",
    "SFTExample",
    "build_model",
    "collate_fn",
    "get_device",
    "train",
]


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


def collate_fn(
    examples: list[SFTExample],
    tokenizer: Any,
    max_length: int,
) -> dict[str, Tensor]:
    sequences = [tokenizer.encode(ex.text, max_length=max_length) for ex in examples]
    sequences = [
        seq if len(seq) >= 2 else seq + [getattr(tokenizer, "pad_id", 0)]
        for seq in sequences
    ]
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
    return NumiForCausalLM(NumiConfig(**cfg["model"]))


def _maybe_apply_lora(model: NumiForCausalLM, cfg: dict[str, Any]) -> bool:
    lora_cfg = cfg.get("lora") or {}
    if not lora_cfg.get("enabled", False):
        return False
    config = LoRAConfig(
        r=int(lora_cfg.get("r", 8)),
        alpha=float(lora_cfg.get("alpha", 16.0)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=tuple(
            lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            )
        ),
    )
    n = apply_lora(model, config)
    mark_only_lora_as_trainable(model)
    print(f"LoRA enabled: replaced {n} Linear layers (r={config.r}, alpha={config.alpha})")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)")
    return True


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

    tokenizer = load_tokenizer(
        config["data"].get("tokenizer_path"),
        vocab_size=config["model"]["vocab_size"],
    )
    if tokenizer.vocab_size != config["model"]["vocab_size"]:
        print(
            f"Adjusting model vocab_size {config['model']['vocab_size']} "
            f"→ {tokenizer.vocab_size} to match tokenizer"
        )
        config["model"]["vocab_size"] = tokenizer.vocab_size

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer, config["data"]["max_seq_length"]
        ),
    )

    model = build_model(config).to(device)
    use_lora = _maybe_apply_lora(model, config)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
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
    print(f"Tokenizer: {type(tokenizer).__name__} (vocab={tokenizer.vocab_size})")
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
                    trainable_params, config["training"]["max_grad_norm"]
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
                    save_checkpoint(
                        ckpt,
                        model=model,
                        config=config,
                        step=global_step,
                        optimizer=optimizer,
                        lora_only=use_lora,
                        extra={"tokenizer_path": config["data"].get("tokenizer_path")},
                    )
                    print(f"Saved {ckpt}")

    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        model=model,
        config=config,
        step=global_step,
        optimizer=optimizer,
        lora_only=use_lora,
        extra={"tokenizer_path": config["data"].get("tokenizer_path")},
    )
    print(f"Training complete → {final_path}")
    return final_path
