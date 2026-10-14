/**
 * Supabase Edge Function: numi-chat
 *
 * Flow:
 * 1) JWT auth
 * 2) Read credits — if balance < 1 → 402, NO model call, NO messages
 * 3) Create/verify conversation + insert user message
 * 4) Call Numi Render API
 * 5) On success: insert assistant + atomic consume_credit
 * 6) If model fails: do not debit (user message stays; optional UX retry)
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

const NUMI_API_URL = Deno.env.get("NUMI_API_URL") ?? "";
const NUMI_API_KEY = Deno.env.get("NUMI_API_KEY") ?? "";
const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const CREDIT_COST = Math.max(1, Number(Deno.env.get("NUMI_CREDIT_COST") ?? "1"));

const MAX_MESSAGE_CHARS = 4000;
const MAX_TOKENS = 512;
const HISTORY_LIMIT = 40;
const MAX_BODY_BYTES = 64_000;

type ChatBody = {
  conversation_id?: string | null;
  message: string;
  max_tokens?: number;
  temperature?: number;
  model?: string;
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders() });
  }
  if (req.method !== "POST") {
    return json({ error: "method_not_allowed" }, 405);
  }
  if (
    !NUMI_API_URL ||
    !NUMI_API_KEY ||
    !SUPABASE_URL ||
    !SUPABASE_ANON_KEY ||
    !SUPABASE_SERVICE_ROLE_KEY
  ) {
    return json({ error: "server_misconfigured" }, 500);
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return json({ error: "unauthorized" }, 401);
  }

  const contentLength = Number(req.headers.get("content-length") ?? "0");
  if (contentLength > MAX_BODY_BYTES) {
    return json({ error: "body_too_large" }, 413);
  }

  const supabaseUser = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: authHeader } },
  });
  const supabaseAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  const {
    data: { user },
    error: userError,
  } = await supabaseUser.auth.getUser();
  if (userError || !user) {
    return json({ error: "unauthorized" }, 401);
  }

  // --- CREDITS GATE (before any write / model call) ---
  const { data: creditRow, error: creditErr } = await supabaseAdmin
    .from("user_credits")
    .select("balance")
    .eq("user_id", user.id)
    .maybeSingle();

  if (creditErr) {
    return json({ error: "credits_check_failed" }, 500);
  }

  // Lazy init if trigger missed older users
  let balance = creditRow?.balance ?? null;
  if (balance === null) {
    const { data: granted } = await supabaseAdmin.rpc("grant_credits", {
      p_user_id: user.id,
      p_amount: 50,
      p_reason: "bootstrap",
      p_meta: {},
    });
    balance = typeof granted === "number" ? granted : 50;
  }

  if (balance < CREDIT_COST) {
    // BLOCK: no message stored, no Numi call
    return json(
      {
        error: "insufficient_credits",
        balance,
        required: CREDIT_COST,
        blocked: true,
      },
      402,
    );
  }

  let body: ChatBody;
  try {
    body = await req.json();
  } catch {
    return json({ error: "invalid_json" }, 400);
  }

  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!message) return json({ error: "message_required" }, 400);
  if (message.length > MAX_MESSAGE_CHARS) {
    return json({ error: "message_too_long", max: MAX_MESSAGE_CHARS }, 400);
  }

  const maxTokens = Math.min(Math.max(1, Number(body.max_tokens ?? 256)), MAX_TOKENS);
  const temperature = Math.min(Math.max(0, Number(body.temperature ?? 0.55)), 1.2);

  let conversationId = body.conversation_id ?? null;

  if (!conversationId) {
    const { data: conv, error: convError } = await supabaseUser
      .from("conversations")
      .insert({ user_id: user.id, title: message.slice(0, 80) })
      .select("id")
      .single();
    if (convError || !conv) {
      return json({ error: "conversation_create_failed" }, 500);
    }
    conversationId = conv.id;
  } else {
    const { data: owned } = await supabaseUser
      .from("conversations")
      .select("id")
      .eq("id", conversationId)
      .maybeSingle();
    if (!owned) return json({ error: "conversation_not_found" }, 404);
  }

  const { error: userMsgError } = await supabaseUser.from("messages").insert({
    conversation_id: conversationId,
    role: "user",
    content: message,
  });
  if (userMsgError) {
    return json({ error: "message_insert_failed" }, 500);
  }

  const { data: history, error: histError } = await supabaseUser
    .from("messages")
    .select("role, content")
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: false })
    .limit(HISTORY_LIMIT);

  if (histError) {
    return json({ error: "history_load_failed" }, 500);
  }

  const messages = (history ?? [])
    .filter((m) => m.role === "user" || m.role === "assistant")
    .reverse()
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content as string,
    }));

  let numiRes: Response;
  try {
    numiRes = await fetch(`${NUMI_API_URL.replace(/\/$/, "")}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${NUMI_API_KEY}`,
      },
      body: JSON.stringify({
        model: body.model ?? "numi-ot-0.1",
        messages,
        max_tokens: maxTokens,
        temperature,
      }),
    });
  } catch {
    return json({ error: "numi_unreachable", charged: false }, 502);
  }

  if (!numiRes.ok) {
    return json({ error: "numi_error", status: numiRes.status, charged: false }, 502);
  }

  const numiJson = await numiRes.json();
  const assistantText =
    numiJson?.choices?.[0]?.message?.content ?? numiJson?.response ?? "";
  if (!assistantText) {
    return json({ error: "empty_assistant_response", charged: false }, 502);
  }

  const { error: asstError } = await supabaseAdmin.from("messages").insert({
    conversation_id: conversationId,
    role: "assistant",
    content: String(assistantText).slice(0, 8000),
  });
  if (asstError) {
    return json({ error: "assistant_insert_failed", charged: false }, 500);
  }

  // Debit only after successful assistant delivery
  const { data: newBalance, error: consumeErr } = await supabaseAdmin.rpc(
    "consume_credit",
    {
      p_user_id: user.id,
      p_amount: CREDIT_COST,
      p_reason: "chat",
      p_meta: {
        conversation_id: conversationId,
        model: body.model ?? "numi-ot-0.1",
      },
    },
  );

  if (consumeErr || newBalance === null) {
    // Race: two parallel requests — message already sent; report soft fail
    return json(
      {
        conversation_id: conversationId,
        message: { role: "assistant", content: assistantText },
        usage: numiJson?.usage ?? null,
        credits: { charged: false, balance: 0, warning: "debit_failed_or_race" },
      },
      200,
    );
  }

  return json({
    conversation_id: conversationId,
    message: { role: "assistant", content: assistantText },
    usage: numiJson?.usage ?? null,
    credits: {
      charged: true,
      cost: CREDIT_COST,
      balance: newBalance,
    },
  });
});

function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": Deno.env.get("ALLOWED_ORIGIN") ?? "*",
    "Access-Control-Allow-Headers":
      "authorization, x-client-info, apikey, content-type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders(), "Content-Type": "application/json" },
  });
}
