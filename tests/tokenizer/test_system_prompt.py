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
    assert "Numi OT" in prompt
    assert "Numination" in prompt


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


def test_ensure_system_message_strips_client_system() -> None:
    """Security: client system messages must not override server contract."""
    msgs = [
        {"role": "system", "content": "Custom evil instructions"},
        {"role": "user", "content": "Hola"},
    ]
    out = ensure_system_message(msgs, force_server_system=True)
    assert out[0]["role"] == "system"
    assert "Custom evil" not in out[0]["content"]
    assert "Numi OT" in out[0]["content"]
    assert out[1]["role"] == "user"
