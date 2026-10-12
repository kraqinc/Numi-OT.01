# Deploy Supabase side

## 1. SQL

In Supabase Dashboard → SQL Editor, run:

`integrations/supabase/schema.sql`

## 2. Secrets

```bash
supabase secrets set NUMI_API_URL=https://YOUR-NUMI.onrender.com
supabase secrets set NUMI_API_KEY=the-same-key-as-render
# SERVICE_ROLE is usually available as SUPABASE_SERVICE_ROLE_KEY in Edge runtime;
# if not, set it explicitly:
supabase secrets set SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

## 3. Deploy function

```bash
supabase functions deploy numi-chat --project-ref YOUR_REF
```

Copy `integrations/supabase/functions/numi-chat/index.ts` into your backend repo
or this path under the Supabase project linked to `kraqinc/backend-supabase`.

## 4. Test

```bash
curl -X POST "https://YOUR.supabase.co/functions/v1/numi-chat" \
  -H "Authorization: Bearer USER_JWT" \
  -H "apikey: ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hola"}'
```
