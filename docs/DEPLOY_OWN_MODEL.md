# Deploy Numi propio (independiente)

Flujo:

```text
Flutter → Supabase Edge → Render (NUMI_BACKEND=peft)
                            ├─ base: SmolLM2-1.7B (HF)
                            └─ adapter: tu LoRA entrenado
```

Sin OpenAI / sin Anthropic.

## 1. Tener el adapter

Desde Colab (después del train):

```bash
# en el repo local/Termux o tras descargar el zip
bash scripts/bundle_adapter_for_deploy.sh /ruta/a/final
# deja adapters/final/ con adapter_config.json + pesos
```

**Importante:** `adapters/` debe estar en el contexto de build de Docker (no ignorear el dir del adapter). El base **no** va en git: se baja de Hugging Face al arrancar.

## 2. Render

1. Web Service → repo `kraqinc/Numi-OT.01` → **Docker**
2. Plan **Standard** o superior (1.7B en CPU necesita RAM; free/starter se queda corto)
3. Env:

| Key | Value |
|-----|--------|
| `NUMI_ENV` | `production` |
| `NUMI_BACKEND` | `peft` |
| `NUMI_API_KEY` | secreto ≥16 chars |
| `NUMI_PEFT_BASE` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| `NUMI_PEFT_ADAPTER` | `/app/adapters/final` |
| `NUMI_PEFT_4BIT` | `0` (CPU) |
| `NUMI_WARMUP` | `1` |
| `HF_TOKEN` | opcional si el base/adapter son privados |

4. Health: `/live` (proceso vivo). Comprueba tú `/ready` → `status: ready`.

## 3. Supabase

```bash
supabase secrets set NUMI_API_URL=https://TU.onrender.com
supabase secrets set NUMI_API_KEY=<mismo que Render>
```

Edge Function `numi-chat` ya habla OpenAI-compatible contigo.

## 4. Probar

```bash
curl -s https://TU.onrender.com/live
curl -s https://TU.onrender.com/ready
curl -s https://TU.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer TU_NUMI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"¿Qué modelo eres?"}]}'
```

Esperado: identidad **Numi OT**.

## Nota de potencia

Esto es **tu** modelo (1.7B + LoRA). No es Fable 5. Para más potencia: base más grande + GPU (RunPod/Lambda) con el mismo `NUMI_BACKEND=peft`.
