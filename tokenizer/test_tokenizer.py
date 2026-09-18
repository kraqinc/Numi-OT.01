from tokenizer import (
    NumiChatTemplate,
    NumiMessage,
    NumiSpecialTokens,
    NumiTokenizerConfig,
)


def test_tokenizer_config() -> None:
    config = NumiTokenizerConfig()

    assert config.name == "Numi-OT.01"
    assert config.base_tokenizer == "Qwen/Qwen3-30B-A3B-Base"
    assert config.revision == "main"


def test_special_tokens() -> None:
    tokens = NumiSpecialTokens()

    assert tokens.bos != tokens.eos
    assert tokens.eos != tokens.pad
    assert tokens.system != tokens.user
    assert tokens.user != tokens.assistant
    assert tokens.assistant != tokens.tool
    assert tokens.tool != tokens.tool_result


def test_message_validation() -> None:
    message = NumiMessage(
        role="user",
        content="Hola Numi",
    )

    assert message.role == "user"
    assert message.content == "Hola Numi"


def test_chat_template_render() -> None:
    tokens = NumiSpecialTokens()

    template = NumiChatTemplate.from_special_tokens(tokens)

    prompt = template.render(
        [
            {
                "role": "system",
                "content": "You are Numi.",
            },
            {
                "role": "user",
                "content": "Hola",
            },
        ]
    )

    assert tokens.bos in prompt
    assert tokens.system in prompt
    assert tokens.user in prompt
    assert tokens.assistant in prompt
    assert tokens.eos in prompt

    assert prompt.endswith(f"{tokens.assistant}\n")


def test_chat_template_without_generation_prompt() -> None:
    tokens = NumiSpecialTokens()

    template = NumiChatTemplate.from_special_tokens(tokens)

    prompt = template.render(
        [
            {
                "role": "user",
                "content": "Prueba",
            }
        ],
        add_generation_prompt=False,
    )

    assert tokens.user in prompt
    assert tokens.eos in prompt

    assert not prompt.endswith(f"{tokens.assistant}\n")


def test_chat_template_with_dataclass_messages() -> None:
    tokens = NumiSpecialTokens()

    template = NumiChatTemplate.from_special_tokens(tokens)

    prompt = template.render(
        [
            NumiMessage(
                role="user",
                content="¿Qué puedes hacer?",
            ),
            NumiMessage(
                role="assistant",
                content="Puedo ayudarte.",
            ),
        ]
    )

    assert "¿Qué puedes hacer?" in prompt
    assert "Puedo ayudarte." in prompt