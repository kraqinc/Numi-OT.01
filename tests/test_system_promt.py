from tokenizer.system_prompt import DEFAULT_NUMI_SYSTEM_PROMPT, NumiSystemPrompt


def test_default_system_prompt() -> None:
    prompt = DEFAULT_NUMI_SYSTEM_PROMPT.render()

    assert "Identity: Numi" in prompt
    assert "Mission:" in prompt
    assert "Personality:" in prompt
    assert "Reasoning:" in prompt
    assert "Communication:" in prompt
    assert "Coding:" in prompt
    assert "Honesty:" in prompt
    assert "Safety:" in prompt


def test_system_prompt_message() -> None:
    prompt = NumiSystemPrompt()

    message = prompt.as_message()

    assert message["role"] == "system"
    assert isinstance(message["content"], str)
    assert len(message["content"]) > 0