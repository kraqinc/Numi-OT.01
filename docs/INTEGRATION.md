# Integración completa Numi OT → App Numination

## Arquitectura final

```text
Flutter (JWT)
    → Supabase Edge Function numi-chat
        → Numi API (Render/Railway)  /v1/chat/completions
            → checkpoint .pt + system prompt forzado
        → Postgres messages (RLS: user solo inserta role=user)
```

**Nunca** pongas el checkpoint ni `NUMI_API_KEY` en Flutter.

---

## Paso 0 — Clonar y smoke train (local)

```bash
git clone https://github.com/kraqinc/Numi-OT.01.git
cd Numi-OT.01
bash scripts/integration/bootstrap_smoke.sh
```

Genera: `outputs/sft_smoke/final.pt` (modelo pequeño, solo para cablear).

Prueba local:

```bash
export NUMI_ENV=development
export NUMI_CONFIG=configs/serving/smoke.yaml
export NUMI_CHECKPOINT=outputs/sft_smoke/final.pt
python scripts/serve.py --host 127.0.0.1 --port 8000

curl -s http://127.0.0.1:8000/live
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hola"}]}'
```

---

## Paso 1 — API en Render

1. New Web Service → repo `kraqinc/Numi-OT.01` → Docker.
2. O usa `render.yaml`.
3. Env (producción):

| Variable | Valor |
|----------|--------|
| `NUMI_ENV` | `production` |
| `NUMI_API_KEY` | secreto ≥16 chars (mismo en Supabase) |
| `NUMI_CONFIG` | `configs/serving/smoke.yaml` |
| `NUMI_CHECKPOINT` | path al `.pt` en el disk/volume |
| `NUMI_DEVICE` | `cpu` |
| `NUMI_CORS_ORIGINS` | origen de tu web si aplica |

4. Health: `/live`
5. Guarda la URL: `https://numi-xxxx.onrender.com`

**Importante:** el smoke checkpoint hay que subirlo al servicio (volume, release artifact o bake en imagen de deploy privado). No está en git (gitignore).

---

## Paso 2 — Supabase

Ver `integrations/supabase/DEPLOY.md`.

Resumen:

1. Ejecutar `schema.sql`
2. Deploy `numi-chat`
3. Secrets: `NUMI_API_URL`, `NUMI_API_KEY`, service role

---

## Paso 3 — Flutter

Copia `integrations/flutter/numi_client.dart` al app repo.

```dart
final client = NumiClient(supabaseUrl: 'https://YOUR.supabase.co');
final res = await client.chat(message: 'Hola');
// res.content, res.conversationId
```

Usuario debe estar logueado (Google/GitHub/email) para tener JWT.

---

## Checklist go-live integración

- [ ] Smoke train OK local (`/v1/chat/completions` responde 200)
- [ ] Render `/live` = alive, `/health` loaded=true
- [ ] Sin `NUMI_API_KEY` → 401 en prod
- [ ] Con key incorrecta → 401
- [ ] Supabase SQL + RLS aplicados
- [ ] Edge Function 200 con JWT válido
- [ ] Flutter recibe `message.content`
- [ ] Usuario no puede insertar `role=assistant` (probado en SQL)

---

## Calidad del modelo

El smoke **no** es un chat de producto. Para calidad real:

1. Ampliar `datasets/2026/`
2. Entrenar `configs/training/sft.yaml` (o mayor) en GPU
3. Sustituir checkpoint en Render
4. Re-test identidad (`evaluation/suites/identity.py`)

---

## Seguridad (ya en código)

- System prompt solo del servidor
- Roles cliente: user|assistant en API schema; assistant en DB solo service role
- Escape de tokens estructurales
- Auth obligatoria en production
- Rate limit + concurrency + input limits
