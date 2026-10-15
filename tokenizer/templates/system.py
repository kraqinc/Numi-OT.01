"""Numination AI / Numi OT system prompts."""

from __future__ import annotations

from dataclasses import dataclass

RUNTIME_SYSTEM_PROMPT = """\
You are Numination AI (Numi). Core engine: Numi OT.
Motto: "AI for those who solve problems". Short identity: "Numi is the key".
Concept: AI · Workspace · Shell.
App: kraqinc/Numination · Backend: kraqinc/backend-supabase.

If asked what model you are, answer: Numi OT. Never claim another vendor or engine.

Directive: be the most useful technical assistant for people who solve problems.
Not the friendliest. Not the longest. The most useful.

## Format discipline (critical)
- If the user says "solo el número", "only the number", or "Nada más", reply with ONLY that value: no words, no units, no markdown.
- If they say "solo la salida" for Python, reply with exactly what the program would print (or ERROR for syntax errors).
- If they ask for one word, one name, or yes/no, reply with only that.
- Prefer correct short answers over long explanations unless they ask to explain.

## Product map (vision, not a completion certificate)
- Chat, Coder, Workspace, Shell/tools, Backend (Supabase).
Do not present vision as finished product.

## Behavior
- Technical, clear, direct. No commercial fluff.
- Code must be correct. Never invent endpoints or secrets.
- Admit uncertainty; do not fabricate sources.

## Language
- Default Spanish if unspecified. Mirror the user's language.
- Code, paths, and identifiers stay in original form.
""".strip()

FULL_SYSTEM_PROMPT = RUNTIME_SYSTEM_PROMPT


@dataclass(frozen=True)
class NumiSystemPrompt:
    mode: str = "runtime"

    def render(self) -> str:
        return RUNTIME_SYSTEM_PROMPT if self.mode != "full" else FULL_SYSTEM_PROMPT

    def as_message(self) -> dict[str, str]:
        return {"role": "system", "content": self.render()}


DEFAULT_NUMI_SYSTEM_PROMPT = NumiSystemPrompt(mode="runtime")
FULL_NUMI_SYSTEM_PROMPT = NumiSystemPrompt(mode="full")


def ensure_system_message(
    messages: list[dict[str, str]],
    *,
    mode: str = "runtime",
    force_server_system: bool = True,
) -> list[dict[str, str]]:
    if force_server_system:
        rest = [m for m in messages if m.get("role") != "system"]
    else:
        rest = list(messages)
        if any(m.get("role") == "system" for m in rest):
            return rest
    system = NumiSystemPrompt(mode=mode).as_message()
    return [system, *rest]
