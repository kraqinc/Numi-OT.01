# Numi-OT.01

**Original sparse Mixture-of-Experts language model for Numination.**

Built from scratch. No weights or architecture copied from external model families.

## Architecture

| Component | Design |
|-----------|--------|
| Attention | GQA + RoPE + SDPA + KV-cache |
| FFN | Sparse MoE (top-k, sorted dispatch, SwiGLU) |
| Norm | RMSNorm |
| Tokenizer | Trainable BPE (`tokenizer/algorithms/bpe.py`) |

## Quickstart

```bash
pip install -r requirements-dev.txt
pytest -q tests/architecture tests/tokenizer tests/training tests/serving tests/evaluation
```

### Train tokenizer (BPE)

```bash
python scripts/tokenizer/train_tokenizer.py \
  --data datasets/2026/numi_conversation_seed.jsonl \
  --output artifacts/tokenizer \
  --vocab-size 4000
```

### SFT

```bash
python scripts/training/train_sft.py --config configs/training/sft.yaml --device cpu
```

### DPO (preferences)

```bash
python scripts/training/train_dpo.py --config configs/training/dpo.yaml --device cpu
```

### Evaluate

```bash
python scripts/evaluation/evaluate.py --device cpu
```

### Generate

```bash
python scripts/generate.py --prompt "Hola" --max-new-tokens 16 --device cpu
```

### API

```bash
python scripts/serve.py --port 8000
# or
docker compose up --build
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"numi-ot-0.1","messages":[{"role":"user","content":"Hola"}],"max_tokens":32}'
```

## Layout

```
architecture/  model/  tokenizer/  training/  serving/
evaluation/  datasets/  configs/  scripts/  tests/
Dockerfile  docker-compose.yml
```

## Status

- [x] MoE transformer + KV-cache + SDPA
- [x] Sorted MoE dispatch
- [x] SFT loop
- [x] Trainable BPE tokenizer
- [x] Minimal DPO
- [x] Smoke evaluation (perplexity)
- [x] OpenAI-compatible API + Docker
- [ ] Large-scale CPT
- [ ] Production tokenizer trained on full corpus

## License

Numination — see repository policy.
