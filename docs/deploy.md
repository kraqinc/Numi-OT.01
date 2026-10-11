# Dónde hospedar Numi y cómo conectarlo a Supabase

## Regla de oro

| Capa | Dónde va |
|------|----------|
| **Flutter app** | Play Store / sideload APK |
| **Supabase** | Auth + Postgres + Edge Functions (ya en la nube de Supabase) |
| **Numi API (modelo)** | Servidor **siempre encendido** con CPU/GPU suficiente |

```text
App → Supabase Edge Function → NUMI_API_URL → Render/Railway/Fly/VPS
```

## Vercel / Netlify — **NO** para el modelo

- Serverless: timeouts cortos (10–60s).
- Sin GPU y poca RAM.
- Cold starts matan la carga de un checkpoint PyTorch.
- Sirven para la **landing** de Numination, no para Numi OT inference.

## Dónde sí poner Numi

### 1) Render (recomendado para empezar, CPU)

1. New → Web Service → conecta `kraqinc/Numi-OT.01`.
2. Runtime: Docker (usa el `Dockerfile` del repo).
3. Instance: al menos **2 GB RAM** (mejor 4 GB) para el config smoke.
4. Env vars:

```env
NUMI_DEVICE=cpu
NUMI_CONFIG=configs/training/sft.yaml
NUMI_CHECKPOINT=outputs/sft/final.pt   # si subes el checkpoint
NUMI_API_KEY=pon-un-secreto-largo
NUMI_MODEL_ID=numi-ot-0.1
PORT=8000
```

5. Health check path: `/health`
6. Copia la URL pública: `https://numi-xxxx.onrender.com`

### 2) Railway

Similar: Docker deploy, set mismas env vars, genera dominio `*.up.railway.app`.

### 3) Fly.io

Bueno si quieres región cerca de tus usuarios; Docker + `fly launch`.

### 4) GPU de verdad (cuando el modelo crezca)

- RunPod / Vast.ai / a VPS con NVIDIA
- `NUMI_DEVICE=cuda` y checkpoint grande

### 5) VPS barato (Hetzner, DigitalOcean)

```bash
git clone ... && docker compose up -d
# nginx + HTTPS delante
```

## Conectar Supabase

En el proyecto Supabase:

```bash
# CLI o dashboard → Edge Function secrets
supabase secrets set NUMI_API_URL=https://numi-xxxx.onrender.com
supabase secrets set NUMI_API_KEY=el-mismo-secreto-que-en-render
```

1. Ejecuta `integrations/supabase/schema.sql`.
2. Deploy `integrations/supabase/functions/numi-chat`.
3. Desde Flutter: llama a `https://<project>.supabase.co/functions/v1/numi-chat` con el JWT del usuario.

La Edge Function **nunca** expone el modelo al cliente; solo Supabase habla con Render.

## Checklist mínimo

- [ ] Numi en Render/Railway con `/health` = ok
- [ ] `NUMI_API_KEY` igual en Render y en secrets de Supabase
- [ ] SQL + RLS aplicados
- [ ] Edge Function `numi-chat` desplegada
- [ ] Flutter usa solo Supabase Functions URL (no la URL de Render)

## Coste mental

| Opción | Modelo pequeño CPU | Modelo grande GPU |
|--------|--------------------|-------------------|
| Vercel | No | No |
| Netlify | No | No |
| Render free/paid | Sí (lento) | No |
| Railway | Sí | Limitado |
| RunPod / VPS GPU | Oversized | Sí |
