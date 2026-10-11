# Connecting Numi to Supabase

Supabase is the **product backend** (Auth + Postgres + Edge Functions).  
Numi is a **separate inference service** (FastAPI).

```text
Client → Supabase (auth, DB, edge fn) → Numi API → GPU/CPU host
```

## 1. Tables

```sql
create table public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text,
  created_at timestamptz not null default now()
);

create table public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('system', 'user', 'assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

alter table public.conversations enable row level security;
alter table public.messages enable row level security;

-- Example policies: users only see their rows
create policy "own conversations" on public.conversations
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "own messages" on public.messages
  for all using (
    exists (
      select 1 from public.conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1 from public.conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  );
```

## 2. Edge Function sketch

Store secrets:

- `NUMI_API_URL` → `https://your-numi-host`
- `NUMI_API_KEY` → optional shared secret

Pseudo-handler:

1. Verify JWT (`Authorization: Bearer ...`).
2. Insert user message into `messages`.
3. `fetch(`${NUMI_API_URL}/v1/chat/completions`, { ... })`.
4. Insert assistant message.
5. Return assistant content to the client.

## 3. Deploy Numi

- Railway / Fly / RunPod / any VPS with Docker:
  - `docker compose up --build`
  - set `NUMI_CHECKPOINT` to your trained weights

## 4. Security

- Never expose the Numi host without auth if it is public.
- Prefer server-side calls from Edge Functions, not from the browser.
