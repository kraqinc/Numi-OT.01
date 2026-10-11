/**
 * Supabase Edge Function: numi-chat
 *
 * Auth (JWT) → persist user message → call Numi API → persist assistant → return.
 *
 * Secrets:
 *   NUMI_API_URL  e.g. https://numi.example.com
 *   NUMI_API_KEY  optional shared secret (sent as Authorization Bearer)
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

const NUMI_API_URL = Deno.env.get("NUMI_API_URL") ?? "";
const NUMI_API_KEY = Deno.env.get("NUMI_API_KEY") ?? "";
const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";

type ChatBody = {
  conversation_id?: string | null;
  message: string;
  max_tokens?: number;
  temperature?: number;
  model?: string;
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      headers: corsHeaders(),
    });
  }

  if (req.method !== "POST") {
    return json({ error: "method_not_allowed" }, 405);
  }

  if (!NUMI_API_URL || !SUPABASE_URL || !SUPABASE_ANON_KEY) {
    return json({ error: "server_misconfigured" }, 500);
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return json({ error: "unauthorized" }, 401);
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: authHeader } },
  });

  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  if (userError || !user) {
    return json({ error: "unauthorized", detail: userError?.message }, 401);
  }

  let body: ChatBody;
  try {
    body = await req.json();
  } catch {
    return json({ error: "invalid_json" }, 400);
  }

  const message = (body.message ?? "").trim();
  if (!message) {
    return json({ error: "message_required" }, 400);
  }

  let conversationId = body.conversation_id ?? null;

  // Create conversation if needed
  if (!conversationId) {
    const { data: conv, error: convError } = await supabase
      .from("conversations")
      .insert({
        user_id: user.id,
        title: message.slice(0, 80),
      })
      .select("id")
      .single();

    if (convError || !conv) {
      return json({ error: "conversation_create_failed", detail: convError?.message }, 500);
    }
    conversationId = conv.id;
  } else {
    // Ownership check via RLS; soft-fail if missing
    const { data: owned } = await supabase
      .from("conversations")
      .select("id")
      .eq("id", conversationId)
      .maybeSingle();
    if (!owned) {
      return json({ error: "conversation_not_found" }, 404);
    }
  }

  // Persist user message
  const { error: userMsgError } = await supabase.from("messages").insert({
    conversation_id: conversationId,
    role: "user",
    content: message,
  });
  if (userMsgError) {
    return json({ error: "message_insert_failed", detail: userMsgError.message }, 500);
  }

  // Load recent history (last 20 turns)
  const { data: history, error: histError } = await supabase
    .from("messages")
    .select("role, content")
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: true })
    .limit(40);

  if (histError) {
    return json({ error: "history_load_failed", detail: histError.message }, 500);
  }

  // Numi API injects its own system prompt if none is sent
  const messages = (history ?? []).map((m) => ({
    role: m.role as "user" | "assistant" | "system",
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
        max_tokens: body.max_tokens ?? 256,
        temperature: body.temperature ?? 0.7,
      }),
    });
  } catch (err) {
    return json({
      error: "numi_unreachable",
      detail: err instanceof Error ? err.message : String(err),
    }, 502);
  }

  if (!numiRes.ok) {
    const text = await numiRes.text();
    return json({
      error: "numi_error",
      status: numiRes.status,
      detail: text.slice(0, 500),
    }, 502);
  }

  const numiJson = await numiRes.json();
  const assistantText =
    numiJson?.choices?.[0]?.message?.content ??
    numiJson?.response ??
    "";

  if (!assistantText) {
    return json({ error: "empty_assistant_response", raw: numiJson }, 502);
  }

  const { error: asstError } = await supabase.from("messages").insert({
    conversation_id: conversationId,
    role: "assistant",
    content: assistantText,
  });
  if (asstError) {
    return json({ error: "assistant_insert_failed", detail: asstError.message }, 500);
  }

  return json({
    conversation_id: conversationId,
    message: {
      role: "assistant",
      content: assistantText,
    },
    usage: numiJson?.usage ?? null,
  });
});

function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...corsHeaders(),
      "Content-Type": "application/json",
    },
  });
}
