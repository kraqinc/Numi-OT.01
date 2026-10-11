from __future__ import annotations

import json
from pathlib import Path

import torch

from training.sft.train import (
    ConversationDataset,
    SimpleTokenizer,
    build_model,
    collate_fn,
)


def test_simple_tokenizer_roundtrip_specials() -> None:
    tok = SimpleTokenizer(vocab_size=1024)
    text = "<|numi_bos|>hola<|numi_eos|>"
    ids = tok.encode(text, max_length=64)
    assert len(ids) > 0
    assert tok.special_to_id["<|numi_bos|>"] in ids


def test_conversation_dataset_loads_seed() -> None:
    path = Path("datasets/2026/numi_conversation_seed.jsonl")
    assert path.exists()
    ds = ConversationDataset(path, max_length=256)
    assert len(ds) >= 10
    assert "<|numi_" in ds[0].text or "numi" in ds[0].text.lower() or len(ds[0].text) > 0


def test_collate_shapes() -> None:
    path = Path("datasets/2026/numi_conversation_seed.jsonl")
    ds = ConversationDataset(path, max_length=128)
    tok = SimpleTokenizer(vocab_size=1024)
    batch = collate_fn([ds[0], ds[1]], tok, max_length=128)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape


def test_sft_one_train_step() -> None:
    config = {
        "model": {
            "vocab_size": 512,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "max_position_embeddings": 128,
            "tie_word_embeddings": True,
        },
        "data": {
            "path": "datasets/2026/numi_conversation_seed.jsonl",
            "max_seq_length": 64,
        },
        "training": {
            "output_dir": "outputs/test_sft",
            "num_epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "logging_steps": 100,
            "save_steps": 1000,
            "seed": 0,
            "device": "cpu",
        },
    }

    model = build_model(config)
    model.train()

    ds = ConversationDataset(config["data"]["path"], max_length=64)
    tok = SimpleTokenizer(vocab_size=config["model"]["vocab_size"])
    batch = collate_fn([ds[0]], tok, max_length=64)

    out = model(input_ids=batch["input_ids"], labels=batch["labels"])
    assert out.loss is not None
    out.loss.backward()

    # at least one grad should exist
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)
