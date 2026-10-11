# Connecting Numi to Supabase

Canonical integration lives under `integrations/supabase/`.

```text
Client → Edge Function numi-chat (JWT) → Numi API → Postgres messages
```

## Files

| Path | Role |
|------|------|
| `integrations/supabase/schema.sql` | conversations + messages + RLS |
| `integrations/supabase/functions/numi-chat/index.ts` | Edge Function |
| `integrations/supabase/README.md` | deploy steps |

## Flow

1. Validate user JWT with Supabase Auth.
2. Create or load `conversations` row.
3. Insert user `messages` row.
4. Load recent history.
5. `POST {NUMI_API_URL}/v1/chat/completions`.
6. Insert assistant message.
7. Return content to the client.

Numi injects the **runtime system prompt** when the payload has no `system` role.

## Do not

- Run the LLM inside Edge Functions.
- Put shared API keys in the Flutter client.
- Disable RLS on chat tables.
