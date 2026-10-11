# Numi-OT.01

**Original sparse Mixture-of-Experts language model for Numination.**

Built from scratch. Independent tokenizer, architecture, training and serving stack.

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
# Dev + tests + API
pip install -r requirements-dev.txt
pip install -e .

# Or CI-style (CPU torch first)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ci.txt
pip install -e .
```

## Quick commands

```bash
# Tokenizer
python scripts/tokenizer/train_tokenizer.py \
  --data datasets/2026/numi_conversation_seed.jsonl \
  --output artifacts/tokenizer --vocab-size 4000

# CPT
python scripts/training/train_cpt.py --config configs/training/cpt.yaml --device cpu --max-steps 20

# SFT
python scripts/training/train_sft.py --config configs/training/sft.yaml --device cpu

# DPO
python scripts/training/train_dpo.py --config configs/training/dpo.yaml --device cpu

# Eval
python scripts/evaluation/evaluate.py --device cpu

# Generate
python scripts/generate.py --prompt "Hola" --device cpu

# API
python scripts/serve.py --port 8000

# Docker
docker compose up --build

# Tests
pytest -q
```

## Docs

| Guide | Path |
|-------|------|
| Architecture | [docs/architecture/README.md](docs/architecture/README.md) |
| Training | [docs/training/README.md](docs/training/README.md) |
| Data formats | [docs/data/README.md](docs/data/README.md) |
| Serving / API | [docs/serving/README.md](docs/serving/README.md) |
| Supabase integration | [docs/supabase.md](docs/supabase.md) |

## Layout

```text
architecture/  model/  tokenizer/  training/  serving/
evaluation/  datasets/  configs/  scripts/  tests/  docs/
```

## Status

- [x] MoE + SDPA + KV-cache
- [x] Trainable BPE (SFT/API/generate)
- [x] CPT streaming
- [x] SFT + DPO
- [x] Smoke eval
- [x] OpenAI-compatible API + Docker
- [x] CI workflows (CPU torch)
- [ ] Full-corpus CPT
- [ ] Production-scale training

## License

Numination — see repository policy.
