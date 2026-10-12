# Numination app-ready (tipo ChatGPT)

## Arquitectura

```text
Flutter  →  Supabase Edge `numi-chat` (JWT)
                →  Numi API (Render)  NUMI_BACKEND=cloud
                      →  Provider OpenAI-compatible
                         (OpenAI / xAI Grok / DeepSeek / Groq / …)
                      →  System prompt Numi OT forzado
```

- **No Anthropic**
- **No pesos en el teléfono**
- Identidad de producto: **Numi OT / Numination**

## Env en Render (producción)

| Variable | Ejemplo |
|----------|---------|
| `NUMI_ENV` | `production` |
| `NUMI_BACKEND` | `cloud` |
| `NUMI_API_KEY` | secreto para Supabase→Numi |
| `NUMI_CLOUD_API_KEY` | key del provider |
| `NUMI_CLOUD_BASE_URL` | `https://api.openai.com/v1` o `https://api.x.ai/v1` |
| `NUMI_CLOUD_MODEL` | `gpt-4.1` / `grok-3` / `deepseek-chat` |
| `NUMI_MODEL_ID` | `numi-ot-0.1` |
| `NUMI_CORS_ORIGINS` | tu web si aplica |

### Ejemplos de provider (elige uno)

**OpenAI**
```text
NUMI_CLOUD_BASE_URL=https://api.openai.com/v1
NUMI_CLOUD_MODEL=gpt-4.1
```

**xAI Grok**
```text
NUMI_CLOUD_BASE_URL=https://api.x.ai/v1
NUMI_CLOUD_MODEL=grok-3
```

**DeepSeek**
```text
NUMI_CLOUD_BASE_URL=https://api.deepseek.com/v1
NUMI_CLOUD_MODEL=deepseek-chat
```

## Supabase

Igual que antes: `schema.sql` + Edge Function + secrets:

```bash
NUMI_API_URL=https://tu-numi.onrender.com
NUMI_API_KEY=<mismo que Render>
```

Flutter solo llama a Supabase (`numi_client.dart`).

## Checklist ready

- [ ] Render con `NUMI_BACKEND=cloud` + cloud key
- [ ] `/live` 200
- [ ] `/v1/chat/completions` con `Authorization: Bearer NUMI_API_KEY` responde como Numi
- [ ] Edge Function desplegada
- [ ] Flutter login + chat
