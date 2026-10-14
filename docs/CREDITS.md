# Créditos Numi ↔ Supabase

**Project URL:** `https://whxqciwphwgzcshejpty.supabase.co`

## Flujo (bloquea si no hay crédito)

```text
Flutter (JWT)
  → POST /functions/v1/numi-chat
       1. ¿balance >= cost?  NO → 402 insufficient_credits  (NO llama al modelo)
       2. Sí → guarda user msg → Render Numi → assistant msg → consume_credit
```

## Rutas que implementas

| Método | Ruta completa | Uso |
|--------|----------------|-----|
| `POST` | `https://whxqciwphwgzcshejpty.supabase.co/functions/v1/numi-chat` | Chat (gate de créditos) |
| `GET`  | `https://whxqciwphwgzcshejpty.supabase.co/functions/v1/numi-credits` | Consultar balance |
| SQL    | `public.user_credits` | Tabla balance |
| RPC    | `consume_credit(p_user_id, p_amount, p_reason, p_meta)` | Débito atómico (service role) |
| RPC    | `grant_credits(...)` | Recargar (webhook pago / admin) |

### Headers Flutter → Edge

```http
Authorization: Bearer <user_access_token>
apikey: <SUPABASE_ANON_KEY>
Content-Type: application/json
```

### Body chat

```json
{
  "message": "Hola",
  "conversation_id": null,
  "max_tokens": 256,
  "temperature": 0.55
}
```

### Respuesta OK (200)

```json
{
  "conversation_id": "uuid",
  "message": { "role": "assistant", "content": "..." },
  "credits": { "charged": true, "cost": 1, "balance": 49 }
}
```

### Sin créditos (402) — **bloquea señal**

```json
{
  "error": "insufficient_credits",
  "balance": 0,
  "required": 1,
  "blocked": true
}
```

En Flutter: si `statusCode == 402` → no muestres burbuja assistant; UI “Sin créditos”.

## Deploy SQL

1. Supabase Dashboard → SQL → pega `integrations/supabase/schema.sql`
2. Deploy functions:

```bash
supabase functions deploy numi-chat --project-ref whxqciwphwgzcshejpty
supabase functions deploy numi-credits --project-ref whxqciwphwgzcshejpty

supabase secrets set NUMI_API_URL=https://TU.onrender.com
supabase secrets set NUMI_API_KEY=<tu key Termux>
supabase secrets set NUMI_CREDIT_COST=1
```

## Qué falta para “la cima” (producto)

| Capacidad | Estado |
|----------|--------|
| Identidad Numi + LoRA open | Parcial (entrenar Hunyuan+SFT) |
| API OpenAI-compatible | Listo |
| Auth JWT + RLS | Listo |
| **Gate de créditos 402** | Listo (este PR) |
| Pagos (Stripe/RevenueCat) → `grant_credits` | Pendiente |
| Streaming SSE al cliente | Pendiente |
| GPU Render / RunPod 7B | Pendiente infra |
| RAG noticias en vivo | Parcial (local) |
| Rate limit por usuario (no solo IP) | Pendiente |
| Nivel Fable-5 cerrado | **No alcanzable open-weight** |

Prioridad siguiente: webhook de pago que llame `grant_credits` + UI 402 en Flutter.
