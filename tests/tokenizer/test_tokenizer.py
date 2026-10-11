from tokenizer import NumiSpecialTokens, NumiTokenizerConfig


def test_tokenizer_config() -> None:
    config = NumiTokenizerConfig()

    assert config.name == "Numi-OT.01"
    assert config.base_tokenizer is None
    assert config.version == "0.1.0"
    assert config.model_max_length == 8192


def test_special_tokens() -> None:
    tokens = NumiSpecialTokens()

    assert tokens.system != tokens.user
    assert tokens.user != tokens.assistant
    assert tokens.tool != tokens.tool_result
    assert tokens.bos != tokens.eos
    assert "<|numi_" in tokens.bos
