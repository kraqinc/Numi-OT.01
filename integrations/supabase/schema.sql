-- Numination chat + credits (hardened RLS)

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- Conversations / messages
-- ---------------------------------------------------------------------------
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

drop policy if exists "conversations_own" on public.conversations;
create policy "conversations_own" on public.conversations
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

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

-- ---------------------------------------------------------------------------
-- Credits / wallet
-- ---------------------------------------------------------------------------
create table if not exists public.user_credits (
  user_id uuid primary key references auth.users (id) on delete cascade,
  balance integer not null default 0 check (balance >= 0),
  lifetime_used integer not null default 0 check (lifetime_used >= 0),
  updated_at timestamptz not null default now()
);

create table if not exists public.credit_ledger (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  delta integer not null,
  reason text not null,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists credit_ledger_user_created_idx
  on public.credit_ledger (user_id, created_at desc);

alter table public.user_credits enable row level security;
alter table public.credit_ledger enable row level security;

-- Users can read their own balance / ledger; never update directly from client
drop policy if exists "credits_select_own" on public.user_credits;
create policy "credits_select_own" on public.user_credits
  for select using (auth.uid() = user_id);

drop policy if exists "ledger_select_own" on public.credit_ledger;
create policy "ledger_select_own" on public.credit_ledger
  for select using (auth.uid() = user_id);

-- New auth user → row with starter credits (adjust default)
create or replace function public.handle_new_user_credits()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.user_credits (user_id, balance)
  values (new.id, 50)
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created_credits on auth.users;
create trigger on_auth_user_created_credits
  after insert on auth.users
  for each row execute function public.handle_new_user_credits();

-- Atomic consume: returns new_balance if ok, null if insufficient
create or replace function public.consume_credit(
  p_user_id uuid,
  p_amount integer default 1,
  p_reason text default 'chat',
  p_meta jsonb default '{}'::jsonb
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  new_balance integer;
begin
  if p_amount is null or p_amount < 1 then
    raise exception 'invalid_amount';
  end if;

  update public.user_credits
  set
    balance = balance - p_amount,
    lifetime_used = lifetime_used + p_amount,
    updated_at = now()
  where user_id = p_user_id
    and balance >= p_amount
  returning balance into new_balance;

  if new_balance is null then
    return null; -- insufficient
  end if;

  insert into public.credit_ledger (user_id, delta, reason, meta)
  values (p_user_id, -p_amount, p_reason, p_meta);

  return new_balance;
end;
$$;

revoke all on function public.consume_credit(uuid, integer, text, jsonb) from public;
grant execute on function public.consume_credit(uuid, integer, text, jsonb) to service_role;

-- Grant credits (admin / webhook)
create or replace function public.grant_credits(
  p_user_id uuid,
  p_amount integer,
  p_reason text default 'grant',
  p_meta jsonb default '{}'::jsonb
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  new_balance integer;
begin
  if p_amount is null or p_amount < 1 then
    raise exception 'invalid_amount';
  end if;

  insert into public.user_credits (user_id, balance)
  values (p_user_id, p_amount)
  on conflict (user_id) do update
    set balance = public.user_credits.balance + excluded.balance,
        updated_at = now()
  returning balance into new_balance;

  insert into public.credit_ledger (user_id, delta, reason, meta)
  values (p_user_id, p_amount, p_reason, p_meta);

  return new_balance;
end;
$$;

revoke all on function public.grant_credits(uuid, integer, text, jsonb) from public;
grant execute on function public.grant_credits(uuid, integer, text, jsonb) to service_role;
