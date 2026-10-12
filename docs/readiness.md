# ¿Está Numi listo para la app?

## Respuesta corta

**No todavía como producto de chat de calidad.**  
**Sí el stack** (código, API, Supabase, seguridad base) para *integrar y probar* un smoke model.

## Qué sí está listo

| Pieza | Estado |
|-------|--------|
| Arquitectura MoE + KV-cache | Código listo |
| Tokenizer + chat template | Listo |
| SFT / LoRA / DPO scripts | Listos |
| API OpenAI-compatible + auth | Lista |
| Edge Function + RLS | Lista (deploy manual) |
| Docker | Listo |
| Hardening VULN-01..07 | En main |

## Qué NO está listo

1. **No hay checkpoint entrenado en el repo** (`outputs/` vacío / gitignored).
2. **Config SFT es toy**: 4 capas, hidden 512, ~decenas de ejemplos seed → no habla como un LLM útil.
3. **Sin corpus grande** → no se puede “subir el modelo a la app” esperando respuestas buenas.
4. **Deploy**: hay que poner la API en Render/Railway + secrets en Supabase.
5. **Flutter** debe llamar solo a `functions/v1/numi-chat`, no a pesos locales.

## Camino mínimo a “demo en la app”

```bash
# 1) Entrena smoke (CPU, minutos)
python scripts/training/train_sft.py --config configs/training/sft.yaml --device cpu

# 2) Sube API a Render con:
#    NUMI_ENV=production
#    NUMI_API_KEY=<secreto>
#    NUMI_CHECKPOINT=outputs/sft/final.pt

# 3) Supabase: schema.sql + deploy numi-chat + secrets
#    NUMI_API_URL, NUMI_API_KEY, SERVICE_ROLE_KEY

# 4) Flutter → POST /functions/v1/numi-chat con JWT
```

## Camino a “modelo de verdad”

- Corpus CPT/SFT de cientos de miles de ejemplos (ES+EN+código)
- Config mayor (o fine-tune de un base abierto con LoRA, sin Qwen identity)
- GPU (RunPod / local)
- Eval de identidad y calidad antes de producción
