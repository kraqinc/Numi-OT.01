# Numi-OT.01

**Original sparse Mixture-of-Experts language model for Numination.**

Built from scratch. No weights or architecture copied from external model families.

## Architecture

| Component | Design |
|-----------|--------|
| Attention | GQA + RoPE + **SDPA** (Flash when available) + KV-cache |
| FFN | Sparse MoE (top-k routing, **sorted dispatch**, SwiGLU experts) |
| Norm | RMSNorm |
| Residual | Pre-norm (Llama-style) |

Default research config (`configs/training/sft.yaml`):

- 4 layers · hidden 512 · 8 heads (GQA 2 KV) · 4 experts (top-2)
- Context 2048 · vocab 128256

Full production-oriented defaults live in `configs/model/numi.yaml` (24 layers, 2048 hidden, 8 experts).

## Project layout

```
architecture/     # attention, moe, rope, rmsnorm, blocks
model/            # NumiConfig, NumiModel, NumiForCausalLM
tokenizer/        # special tokens, chat template, system prompt
training/sft/     # supervised fine-tuning loop
serving/          # FastAPI OpenAI-compatible server
scripts/          # CLI entrypoints
datasets/         # conversation seed and future corpora
configs/          # YAML configs
tests/            # unit tests
```

## Quickstart

### Install

```bash
pip install -r requirements-dev.txt
```

### Run tests

```bash
pytest -q tests/architecture tests/tokenizer tests/inference tests/training tests/serving
```

### Supervised fine-tuning

```bash
python scripts/training/train_sft.py --config configs/training/sft.yaml --device cpu
```

### Generate

```bash
python scripts/generate.py --prompt "Hola" --max-new-tokens 16 --device cpu

python scripts/generate.py \
  --checkpoint outputs/sft/final.pt \
  --prompt "¿Qué puedes hacer?" \
  --max-new-tokens 32
```

### API server (OpenAI-compatible)

```bash
# optional: export NUMI_CHECKPOINT=outputs/sft/final.pt
python scripts/serve.py --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "numi-ot-0.1",
    "messages": [{"role": "user", "content": "Hola"}],
    "max_tokens": 32
  }'
```

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/chat`

## Chat format

```
<|numi_bos|>
<|numi_system|>
{system}
<|numi_eos|>
<|numi_user|>
{user}
<|numi_eos|>
<|numi_assistant|>
```

## Status

- [x] Core MoE transformer
- [x] KV-cache + `generate()`
- [x] SDPA attention path
- [x] Sorted MoE dispatch
- [x] Tokenizer contract + chat template
- [x] Minimal SFT loop
- [x] OpenAI-compatible API
- [ ] Trained tokenizer artifact
- [ ] Full CPT on FineWeb-scale data
- [ ] Alignment (DPO/ORPO)

## License

Numination — see repository policy.
