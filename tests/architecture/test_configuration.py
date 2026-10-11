from pathlib import Path

from model.configs.numi_config import NumiConfig


def test_project_structure() -> None:
    assert Path("architecture").is_dir()
    assert Path("configs").is_dir()
    assert Path("model").is_dir()
    assert Path("tokenizer").is_dir()
    assert Path("tests").is_dir()


def test_numi_config_defaults() -> None:
    config = NumiConfig()

    assert config.vocab_size == 128256
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 4
    assert config.num_experts == 8
    assert config.num_experts_per_tok == 2
    assert config.head_dim == 128


def test_numi_config_validation() -> None:
    try:
        NumiConfig(hidden_size=100, num_attention_heads=16)
        raised = False
    except ValueError:
        raised = True

    assert raised
