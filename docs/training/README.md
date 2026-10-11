# Training

## Order

```text
1. Train BPE tokenizer
2. CPT (optional)
3. SFT (full or LoRA)
4. DPO
5. Evaluate
```

## Tokenizer

```bash
python scripts/tokenizer/train_tokenizer.py \
  --data datasets/2026/numi_conversation_seed.jsonl \
  --output artifacts/tokenizer \
  --vocab-size 4000
```

## CPT

```bash
python scripts/training/train_cpt.py \
  --config configs/training/cpt.yaml \
  --device cpu --max-steps 20
```

## SFT (full fine-tune)

```bash
python scripts/training/train_sft.py --config configs/training/sft.yaml --device cpu
```

## SFT with LoRA

```bash
python scripts/training/train_sft.py --config configs/training/sft_lora.yaml --device cpu
```

LoRA settings live under `lora:` in the YAML:

```yaml
lora:
  enabled: true
  r: 8
  alpha: 16.0
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

Only adapter weights are trained and saved when `lora.enabled: true`.

## DPO

```bash
python scripts/training/train_dpo.py --config configs/training/dpo.yaml --device cpu
```

## Checkpoints

```python
from training.checkpointing import save_checkpoint, load_checkpoint
```

- Full model: `lora_only=False` (default when LoRA is off)
- Adapters only: `lora_only=True` when LoRA is enabled
