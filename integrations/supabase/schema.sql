-- Numination chat persistence for Numi OT (hardened RLS)

create extension if not exists "pgcrypto";

create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  title text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations (id) on delete cascade,
  role text not null check (role in ('system', 'user', 'assistant')),
  content text not null check (char_length(content) <= 8000),
  created_at timestamptz not null default now()
);

create index if not exists messages_conversation_created_idx
  on public.messages (conversation_id, created_at);

create index if not exists conversations_user_updated_idx
  on public.conversations (user_id, updated_at desc);

alter table public.conversations enable row level security;
alter table public.messages enable row level security;

-- Conversations: owner full access
drop policy if exists "conversations_own" on public.conversations;
create policy "conversations_own" on public.conversations
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- Messages: clients may only INSERT role=user; SELECT own conversation history
drop policy if exists "messages_own" on public.messages;
drop policy if exists "messages_select_own" on public.messages;
drop policy if exists "messages_insert_user_only" on public.messages;

create policy "messages_select_own" on public.messages
  for select
  using (
    exists (
      select 1 from public.conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  );

create policy "messages_insert_user_only" on public.messages
  for insert
  with check (
    role = 'user'
    and exists (
      select 1 from public.conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  );

-- No UPDATE/DELETE for clients on messages (server uses service role)

create or replace function public.touch_conversation()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  update public.conversations
  set updated_at = now()
  where id = new.conversation_id;
  return new;
end;
$$;

drop trigger if exists messages_touch_conversation on public.messages;
create trigger messages_touch_conversation
  after insert on public.messages
  for each row execute function public.touch_conversation();

-- Server-side insert for assistant/system via service role (bypass RLS)
-- Edge Function must use SUPABASE_SERVICE_ROLE_KEY for assistant rows.
