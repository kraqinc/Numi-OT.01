# Deploy Numi propio (sin OpenAI / sin Grok API)

Tu clave es **`NUMI_API_KEY`** (la que generaste en Termux).  
Esa key **solo** autentica App → Supabase → **tu** API en Render.  
**No** es una key de OpenAI ni de xAI.

```text
Flutter → Supabase (JWT usuario)
            → Render  Authorization: Bearer NUMI_API_KEY
                 NUMI_BACKEND=peft
                 base: SmolLM2-1.7B (HF)
                 adapter: tu LoRA
```

## Env en Render (solo esto)

| Key | Value |
|-----|--------|
| `NUMI_ENV` | `production` |
| `NUMI_BACKEND` | `peft` |
| `NUMI_API_KEY` | **la misma que en Termux** (≥16 chars) |
| `NUMI_PEFT_BASE` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| `NUMI_PEFT_ADAPTER` | `/app/adapters/final` o `usuario/repo-lora` en HF |
| `NUMI_WARMUP` | `1` |
| `HF_TOKEN` | solo si el adapter en HF es privado |

**No configures** `NUMI_CLOUD_*` ni `OPENAI_API_KEY`.

## Supabase secrets

```bash
supabase secrets set NUMI_API_URL=https://TU.onrender.com
supabase secrets set NUMI_API_KEY=<exactamente la de Termux/Render>
```

## Probar

```bash
curl -s https://TU.onrender.com/live
curl -s https://TU.onrender.com/ready
curl -s https://TU.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer TU_NUMI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"¿Qué modelo eres?"}]}'
```
