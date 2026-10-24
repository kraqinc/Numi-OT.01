# Numi-OT.01

Original sparse Mixture-of-Experts language model for **Numination**.

Built from scratch. No weights or architecture copied from external base models (Qwen, Llama, etc.) for the core Numi path.

## Quick start

```bash
git clone https://github.com/kraqinc/Numi-OT.01.git
cd Numi-OT.01
pip install -e .
```

### Train BPE tokenizer (required for readable text)

```bash
PYTHONPATH=. python scripts/tokenizer/train_tokenizer.py \
  --data datasets/2026/numi_conversation_seed.jsonl \
  --output artifacts/tokenizer \
  --vocab-size 8000
```

### SFT (from-scratch / LoRA)

```bash
# Colab T4 friendly
PYTHONPATH=. python scripts/training/train_sft.py \
  --config configs/training/sft_colab_t4.yaml \
  --device cuda

# Continue from checkpoint
PYTHONPATH=. python scripts/training/train_sft.py \
  --config configs/training/sft_colab_continue.yaml \
  --device cuda
```

### Branch for large architecture

- `main` — base Numi-OT.01
- `numi-largeot` — Large config (~14.7B total / ~29GB bf16) + Colab T4 path + curriculum

## Architecture

- Sparse MoE (SwiGLU experts) + GQA + RoPE + RMSNorm
- Optional LoRA (own implementation, no peft required for core path)
- Chat template + special tokens owned by Numi

## Important

1. **Train the BPE tokenizer** before expecting readable generation. The byte-level fallback cannot produce coherent language with the current data scale.
2. Configs under `configs/training/lora_frontier*.yaml` that reference external `model_id` (Qwen, etc.) are **legacy paths** — not the own-weights path.
3. See `docs/AUDIT_ISSUES.md` for known issues and fixes.

## Docs

- `docs/` — training, deploy, Colab, knowledge
- `knowledge/curriculum/OWN_WEIGHTS_ADVANCED.md` — full own-weights curriculum
- `MODEL_CARD.md` — model card
