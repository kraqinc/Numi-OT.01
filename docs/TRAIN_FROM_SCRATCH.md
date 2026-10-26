# Entrenar Numi desde cero (sin Qwen ni bases externas)

## Objetivo

Pesos propios + BPE propio. Loss que baje de forma real (hacia ~1–2 en overfit de seeds).

## Orden obligatorio

### 1. BPE

```bash
PYTHONPATH=. python scripts/tokenizer/train_tokenizer.py \
  --data datasets/2026/numi_conversation_seed.jsonl \
  --output artifacts/tokenizer \
  --vocab-size 8000
```

Sin esto el modelo no genera texto legible.

### 2. SFT base (T4)

```bash
PYTHONPATH=. python scripts/training/train_sft.py \
  --config configs/training/sft_colab_t4.yaml \
  --device cuda
```

### 3. Overfit (bajar loss fuerte en los mismos seeds)

```bash
PYTHONPATH=. python scripts/training/train_sft.py \
  --config configs/training/sft_overfit.yaml \
  --device cuda
```

Esperado: loss 5.x → 3.x → 2.x → ~1.x si el pipeline está sano.

### 4. Continue con más datos

```bash
PYTHONPATH=. python scripts/training/train_sft.py \
  --config configs/training/sft_colab_continue.yaml \
  --device cuda
```

## Qué NO usar para own-weights

- `configs/training/lora_frontier*.yaml` (referencian Qwen / otros)
- Cualquier `base.model_id` de HuggingFace como peso inicial de Numi

## Notas

- `train.py` alinea `model.vocab_size` al tokenizer y activa bf16 autocast si `bf16: true`.
- LoRA es implementación propia del repo (`training/lora/`).
- Checkpoint en `outputs/.../final.pt`.
