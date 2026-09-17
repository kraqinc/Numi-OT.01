from tokenizer import (
    NumiSpecialTokens,
    NumiTokenizerConfig,
)


def test_tokenizer_config():
    config = NumiTokenizerConfig()

    assert config.name == "Numi-OT.01"
    assert config.base_tokenizer == "Qwen/Qwen3-30B-A3B-Base"


def test_special_tokens():
    tokens = NumiSpecialTokens()

    assert tokens.system != tokens.user
    assert tokens.user != tokens.assistant
    assert tokens.tool != tokens.tool_result