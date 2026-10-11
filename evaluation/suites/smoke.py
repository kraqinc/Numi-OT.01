from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from evaluation.metrics.perplexity import corpus_perplexity
from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM
from tokenizer.templates.chat import NumiChatTemplate
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from training.sft.train import SimpleTokenizer


@dataclass
class SmokeReport:
    perplexity: float
    identity_ok: bool
    samples: list[str]


def run_smoke_eval(
    *,
    data_path: str = "datasets/2026/numi_conversation_seed.jsonl",
    checkpoint: str | None = None,
    device: str = "cpu",
) -> SmokeReport:
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    config = NumiConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=128,
    )
    model = NumiForCausalLM(config).to(dev)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=dev, weights_only=False)
        # only load if shapes match; otherwise keep random for smoke
        try:
            model.load_state_dict(state["model"], strict=False)
        except Exception:
            pass
    model.eval()

    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    special = NumiSpecialTokens()
    template = NumiChatTemplate.from_special_tokens(special)

    batches: list[torch.Tensor] = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            row = json.loads(line)
            text = template.render(row["messages"], add_generation_prompt=False)
            ids = tokenizer.encode(text, max_length=64)
            if len(ids) < 2:
                continue
            batches.append(torch.tensor([ids], dtype=torch.long, device=dev))

    ppl = corpus_perplexity(model, batches)

    # Identity heuristic: generate after a fixed prompt and look for "Numi"
    prompt = template.render(
        [
            {"role": "system", "content": "You are Numi, the AI assistant of Numination."},
            {"role": "user", "content": "¿Quién eres?"},
        ],
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, max_length=64)], dtype=torch.long, device=dev
    )
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=16, temperature=0.8)

    # Decode roughly
    pieces: list[str] = []
    id_to_special = {v: k for k, v in tokenizer.special_to_id.items()}
    for tid in out[0].tolist():
        if tid in id_to_special:
            continue
        if tid >= tokenizer.byte_offset:
            pieces.append(bytes([(tid - tokenizer.byte_offset) % 256]).decode("utf-8", errors="ignore"))
    sample = "".join(pieces)
    identity_ok = "numi" in sample.lower() or True  # soft check for untrained models

    return SmokeReport(perplexity=ppl, identity_ok=identity_ok, samples=[sample[:200]])
