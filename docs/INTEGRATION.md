# Integración Numi (producto real)

## Stack recomendado (2026)

```text
Flutter → Supabase numi-chat → Numi API
                              └─ BaseLoRAEngine (gpt-oss / DeepSeek / Gemma + LoRA Numi)
```

El MoE toy (`sft_smoke` / `sft_mid`) solo valida código. **Chat de producto = BASE_LORA.md.**

## Train real (GPU)

Ver `docs/BASE_LORA.md`.

```bash
pip install -r requirements-lora.txt
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

## Deploy

1. Adapter `outputs/lora_frontier/final` + base HF id en el servidor (GPU recomendada).
2. `NUMI_API_KEY` + `NUMI_ENV=production`.
3. Supabase secrets → Edge Function.
4. Flutter solo `numi-chat`.

## Seguridad (ya en repo)

Auth obligatoria en prod, system forzado, RLS user-only insert, escape de specials.
