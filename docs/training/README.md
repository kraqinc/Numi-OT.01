# Training

## Order

```text
1. Train BPE tokenizer
2. CPT (optional)
3. SFT
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

SFT / CPT / API auto-load `artifacts/tokenizer` via `tokenizer.factory.load_tokenizer`.

## CPT (streaming)

Packs streamed text into fixed-length batches.

```bash
python scripts/training/train_cpt.py \
  --config configs/training/cpt.yaml \
  --device cpu \
  --max-steps 20
```

Config: `configs/training/cpt.yaml`

## SFT

```bash
python scripts/training/train_sft.py \
  --config configs/training/sft.yaml \
  --device cpu
```

- Data: JSONL with `messages` (system/user/assistant)
- Template: Numi chat special tokens
- Checkpoint: `outputs/sft/final.pt`

## DPO

```bash
python scripts/training/train_dpo.py \
  --config configs/training/dpo.yaml \
  --device cpu
```

Preference JSONL fields:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

## Notes

- Start with CPU + tiny config to validate the pipeline.
- Scale `hidden_size` / layers / data only after the smoke path works.
- Always keep `vocab_size` aligned with the active tokenizer.
