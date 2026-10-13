"""LoRA SFT on a frontier open-weight base — real capability + Numi identity."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from training.base_lora.data import ChatJsonlDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate(tokenizer: Any, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        ids = [b["input_ids"] for b in batch]
        padded = tokenizer.pad(
            {"input_ids": ids},
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = padded["input_ids"].clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        padded["labels"] = labels
        return padded

    return collate


def train_lora(config: dict[str, Any]) -> Path:
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "Install peft + transformers + bitsandbytes:\n"
            "  pip install peft bitsandbytes accelerate transformers\n"
            f"Original error: {exc}"
        ) from exc

    base_cfg = config["base"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)

    model_id = base_cfg["model_id"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {model_id}")
    print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=base_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": base_cfg.get("trust_remote_code", False),
        "attn_implementation": base_cfg.get("attn_implementation", "sdpa"),
    }

    dtype_name = base_cfg.get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_name, torch.bfloat16)
    load_kwargs["dtype"] = dtype

    if base_cfg.get("load_in_4bit", True):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except ValueError as exc:
        msg = str(exc).lower()
        if "quantization" in msg or "mxfp4" in msg or "bitsandbytes" in msg:
            print("WARN: base rejects BitsAndBytes; loading without load_in_4bit")
            load_kwargs.pop("quantization_config", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        else:
            raise

    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = ChatJsonlDataset(
        data_cfg["paths"],
        tokenizer=tokenizer,
        max_length=int(data_cfg.get("max_seq_length", 1024)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        shuffle=True,
        collate_fn=_collate(tokenizer, int(data_cfg.get("max_seq_length", 1024))),
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("num_epochs", 2))
    accum = int(train_cfg.get("gradient_accumulation_steps", 8))
    log_every = int(train_cfg.get("logging_steps", 5))
    save_every = int(train_cfg.get("save_steps", 50))
    max_norm = float(train_cfg.get("max_grad_norm", 1.0))

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    micro = 0

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            micro += 1

            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    print(f"epoch={epoch} step={global_step} loss={loss.item() * accum:.4f}")

                if global_step % save_every == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt))
                    tokenizer.save_pretrained(str(ckpt))
                    print(f"Saved {ckpt}")

        # flush leftover grads at end of epoch
        if micro % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print(f"epoch={epoch} step={global_step} (flush) loss={loss.item() * accum:.4f}")
            micro = 0

    final = output_dir / "final"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    meta = {
        "base_model": model_id,
        "lora": lora_cfg,
        "steps": global_step,
        "identity": "Numi OT / Numination AI",
        "seed": seed,
    }
    (final / "numi_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Training complete → {final}")
    return final


def train_from_yaml(path: str | Path) -> Path:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return train_lora(config)
