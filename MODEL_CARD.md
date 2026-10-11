# Model Card — Numi-OT.01

## Model details

- **Name:** Numi-OT.01
- **Org:** Numination
- **Type:** Decoder-only sparse MoE language model (research / product prototype)
- **License:** Proprietary / closed app intent for Numination product surfaces

## Architecture

- GQA + RoPE + RMSNorm + SwiGLU MoE
- KV-cache generation
- Optional LoRA fine-tuning

## Intended use

Technical assistant for Numination (chat, coding help, workspace). Identity: **Numi OT**.

## Out of scope

- Medical / legal advice as authority
- Running without rate limits on public internet
- Claiming third-party vendor identity

## Training data

Seed conversation + identity + preference JSONL under `datasets/2026/`. Not a full web pretrain corpus yet.

## Evaluation

- Unit tests (architecture, tokenizer, training, serving)
- Identity prompt checks (`evaluation/suites/identity.py`)
- Smoke eval suite

## Deployment

See `docs/deploy.md`. Inference API behind Supabase Edge Functions + API key.
