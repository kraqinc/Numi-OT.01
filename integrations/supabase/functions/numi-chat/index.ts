/**
 * Supabase Edge Function: numi-chat (hardened)
 *
 * - User JWT for auth + user message insert (role=user only via RLS)
 * - Service role for assistant insert (VULN-02)
 * - Recent history (desc + reverse) (HALLAZGO-19)
 * - Input limits (VULN-05)
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

const NUMI_API_URL = Deno.env.get("NUMI_API_URL") ?? "";
const NUMI_API_KEY = Deno.env.get("NUMI_API_KEY") ?? "";
const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

const MAX_MESSAGE_CHARS = 4000;
const MAX_TOKENS = 512;
const HISTORY_LIMIT = 40;

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
  if (!NUMI_API_URL || !SUPABASE_URL || !SUPABASE_ANON_KEY || !SUPABASE_SERVICE_ROLE_KEY) {
    return json({ error: "server_misconfigured" }, 500);
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return json({ error: "unauthorized" }, 401);
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

  let body: ChatBody;
  try {
    body = await req.json();
  } catch {
    return json({ error: "invalid_json" }, 400);
  }

  const message = (body.message ?? "").trim();
  if (!message) return json({ error: "message_required" }, 400);
  if (message.length > MAX_MESSAGE_CHARS) {
    return json({ error: "message_too_long", max: MAX_MESSAGE_CHARS }, 400);
  }

  const maxTokens = Math.min(Math.max(1, body.max_tokens ?? 256), MAX_TOKENS);
  const temperature = Math.min(Math.max(0, body.temperature ?? 0.7), 2);

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

  // User message only (RLS enforces role=user)
  const { error: userMsgError } = await supabaseUser.from("messages").insert({
    conversation_id: conversationId,
    role: "user",
    content: message,
  });
  if (userMsgError) {
    return json({ error: "message_insert_failed" }, 500);
  }

  // Recent history: newest first, then reverse for chronological order
  const { data: history, error: histError } = await supabaseUser
    .from("messages")
    .select("role, content")
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: false })
    .limit(HISTORY_LIMIT);

  if (histError) {
    return json({ error: "history_load_failed" }, 500);
  }

  // Never trust client system rows: only user + assistant to the model
  const messages = (history ?? [])
    .filter((m) => m.role === "user" || m.role === "assistant")
    .reverse()
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content as string,
    }));

  const numiHeaders: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (NUMI_API_KEY) {
    numiHeaders["Authorization"] = `Bearer ${NUMI_API_KEY}`;
  }

  let numiRes: Response;
  try {
    numiRes = await fetch(`${NUMI_API_URL.replace(/\/$/, "")}/v1/chat/completions`, {
      method: "POST",
      headers: numiHeaders,
      body: JSON.stringify({
        model: body.model ?? "numi-ot-0.1",
        messages,
        max_tokens: maxTokens,
        temperature,
      }),
    });
  } catch {
    return json({ error: "numi_unreachable" }, 502);
  }

  if (!numiRes.ok) {
    return json({ error: "numi_error", status: numiRes.status }, 502);
  }

  const numiJson = await numiRes.json();
  const assistantText =
    numiJson?.choices?.[0]?.message?.content ?? numiJson?.response ?? "";
  if (!assistantText) {
    return json({ error: "empty_assistant_response" }, 502);
  }

  // Assistant insert via service role only (VULN-02)
  const { error: asstError } = await supabaseAdmin.from("messages").insert({
    conversation_id: conversationId,
    role: "assistant",
    content: String(assistantText).slice(0, 8000),
  });
  if (asstError) {
    return json({ error: "assistant_insert_failed" }, 500);
  }

  return json({
    conversation_id: conversationId,
    message: { role: "assistant", content: assistantText },
    usage: numiJson?.usage ?? null,
  });
});

function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": Deno.env.get("ALLOWED_ORIGIN") ?? "*",
    "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders(), "Content-Type": "application/json" },
  });
}
