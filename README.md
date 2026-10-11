# Numi-OT.01

**Original sparse Mixture-of-Experts language model for Numination.**

## Pipeline

```text
1) Train BPE tokenizer
2) CPT (optional streaming pretrain)
3) SFT on conversations
4) DPO on preferences
5) Evaluate + serve
```

## Install

```bash
pip install -r requirements-dev.txt
```

## Train tokenizer

```bash
python scripts/tokenizer/train_tokenizer.py \
  --data datasets/2026/numi_conversation_seed.jsonl \
  --output artifacts/tokenizer \
  --vocab-size 4000
```

SFT / API / generate auto-load `artifacts/tokenizer` if present.

## CPT (streaming)

```bash
python scripts/training/train_cpt.py --config configs/training/cpt.yaml --device cpu --max-steps 20
```

## SFT

```bash
python scripts/training/train_sft.py --config configs/training/sft.yaml --device cpu
```

## DPO

```bash
python scripts/training/train_dpo.py --config configs/training/dpo.yaml --device cpu
```

## Evaluate

```bash
python scripts/evaluation/evaluate.py --device cpu
```

## Generate

```bash
python scripts/generate.py --prompt "Hola" --device cpu
```

## API / Docker

```bash
python scripts/serve.py --port 8000
# or
docker compose up --build
```

## Tests

```bash
pytest -q tests/architecture tests/tokenizer tests/training tests/serving tests/evaluation
```

## Status

- [x] MoE + SDPA + KV-cache
- [x] Trainable BPE integrated into SFT/API/generate
- [x] CPT streaming
- [x] SFT + DPO
- [x] Smoke eval
- [x] FastAPI + Docker
- [ ] Full-corpus CPT
- [ ] Production-scale training
