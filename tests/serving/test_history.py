from serving.history import estimate_tokens, trim_messages


def test_trim_keeps_system_and_recent() -> None:
    messages = [
        {"role": "system", "content": "You are Numi."},
        {"role": "user", "content": "a" * 400},
        {"role": "assistant", "content": "b" * 400},
        {"role": "user", "content": "hola"},
    ]
    trimmed = trim_messages(messages, max_tokens=50)
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "hola"


def test_estimate_tokens() -> None:
    assert estimate_tokens("abcd") == 1
