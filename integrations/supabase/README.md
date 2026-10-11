# Numi ↔ Supabase

Supabase = Auth + Postgres + Edge Functions.  
Numi API = inference (`/v1/chat/completions`).

```text
Client → Edge Function (JWT) → Numi API → response → messages table
```

## Setup

1. Run `schema.sql` in the Supabase SQL editor.
2. Deploy `functions/numi-chat/index.ts` as an Edge Function named `numi-chat`.
3. Set secrets:

```bash
supabase secrets set NUMI_API_URL=https://your-numi-host
supabase secrets set NUMI_API_KEY=optional-shared-secret
```

4. Call from the app:

```http
POST /functions/v1/numi-chat
Authorization: Bearer <user_jwt>
Content-Type: application/json

{
  "conversation_id": "<uuid or null>",
  "message": "Hola",
  "max_tokens": 128
}
```

## Security

- Never call Numi from the browser with a shared key.
- RLS keeps users on their own rows.
- Numi injects the runtime system prompt if the client omits `system`.
