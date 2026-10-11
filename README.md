# Numi-OT.01

**Original sparse Mixture-of-Experts language model for Numination.**

Built from scratch. No weights or architecture copied from external model families.

## Architecture

| Component | Design |
|-----------|--------|
| Attention | GQA + RoPE + **SDPA** (Flash when available) + KV-cache |
| FFN | Sparse MoE (top-k routing, SwiGLU experts) |
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

Serving (optional, separate stack):

```bash
pip install -r requirements-serve.txt
```

### Run tests

```bash
pytest -q tests/architecture tests/tokenizer tests/inference
```

### Supervised fine-tuning (minimal)

```bash
python scripts/training/train_sft.py --config configs/training/sft.yaml --device cpu
```

Override data / epochs:

```bash
python scripts/training/train_sft.py \
  --data datasets/2026/numi_conversation_seed.jsonl \
  --epochs 1 \
  --batch-size 1 \
  --device cpu
```

### Generate

```bash
# random init (smoke)
python scripts/generate.py --prompt "Hola" --max-new-tokens 16 --device cpu

# from checkpoint
python scripts/generate.py \
  --checkpoint outputs/sft/final.pt \
  --prompt "¿Qué puedes hacer?" \
  --max-new-tokens 32
```

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
- [x] Tokenizer contract + chat template
- [x] Minimal SFT loop
- [ ] Trained tokenizer artifact
- [ ] Full CPT on FineWeb-scale data
- [ ] OpenAI-compatible API
- [ ] Alignment (DPO/ORPO)

## License

Numination — see repository policy.
