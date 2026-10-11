from tokenizer.templates.system import (
    DEFAULT_NUMI_SYSTEM_PROMPT,
    FULL_NUMI_SYSTEM_PROMPT,
    NumiSystemPrompt,
    ensure_system_message,
)


def test_default_system_prompt() -> None:
    prompt = DEFAULT_NUMI_SYSTEM_PROMPT.render()

    assert "Numination AI" in prompt or "Numi" in prompt
    assert "Numi OT" in prompt
    assert "AI for those who solve problems" in prompt
    assert "Numi is the key" in prompt
    assert "kraqinc/Numination" in prompt
    assert "kraqinc/backend-supabase" in prompt
    assert "Spanish" in prompt or "español" in prompt.lower() or "mirror" in prompt.lower()


def test_full_system_prompt() -> None:
    prompt = FULL_NUMI_SYSTEM_PROMPT.render()
    assert "SECTION 0" in prompt
    assert "Numi OT" in prompt
    assert len(prompt) > len(DEFAULT_NUMI_SYSTEM_PROMPT.render())


def test_system_prompt_message() -> None:
    message = NumiSystemPrompt().as_message()
    assert message["role"] == "system"
    assert isinstance(message["content"], str)
    assert len(message["content"]) > 0


def test_ensure_system_message_injects() -> None:
    msgs = [{"role": "user", "content": "Hola"}]
    out = ensure_system_message(msgs)
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert "Numi OT" in out[0]["content"]


def test_ensure_system_message_respects_existing() -> None:
    msgs = [
        {"role": "system", "content": "Custom"},
        {"role": "user", "content": "Hola"},
    ]
    out = ensure_system_message(msgs)
    assert out[0]["content"] == "Custom"
    assert len(out) == 2
