import torch

from architecture.embeddings.token import TokenEmbedding
from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM, NumiModel


def _tiny_config(**kwargs) -> NumiConfig:
    defaults = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    defaults.update(kwargs)
    return NumiConfig(**defaults)


def test_configuration_defaults() -> None:
    config = NumiConfig()

    assert config.vocab_size == 128256
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 4
    assert config.num_experts == 8
    assert config.num_experts_per_tok == 2


def test_embedding_shape() -> None:
    embedding = TokenEmbedding(vocab_size=100, hidden_size=32)
    input_ids = torch.tensor([[1, 2, 3]])
    output = embedding(input_ids)
    assert output.shape == (1, 3, 32)


def test_embedding_resize_preserves_old_weights() -> None:
    embedding = TokenEmbedding(vocab_size=10, hidden_size=8)
    original = embedding.weight.detach().clone()

    embedding.resize(13)

    assert embedding.vocab_size == 13
    assert embedding.weight.shape == (13, 8)
    assert torch.equal(embedding.weight[:10], original)


def test_model_forward_shape() -> None:
    config = _tiny_config()
    model = NumiModel(config)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    hidden, cache = model(input_ids, use_cache=False)

    assert hidden.shape == (1, 4, config.hidden_size)
    assert cache is None


def test_causal_lm_forward() -> None:
    config = _tiny_config()
    model = NumiForCausalLM(config)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    output = model(input_ids=input_ids, labels=input_ids)

    assert output.logits is not None
    assert output.loss is not None
    assert output.logits.shape == (1, 4, config.vocab_size)


def test_tied_embeddings() -> None:
    config = _tiny_config(tie_word_embeddings=True)
    model = NumiForCausalLM(config)

    assert (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    )


def test_kv_cache_generation_step() -> None:
    config = _tiny_config()
    model = NumiForCausalLM(config)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    # Prefill
    out1 = model(input_ids, use_cache=True)
    assert out1.past_key_values is not None
    assert len(out1.past_key_values) == config.num_hidden_layers
    assert out1.past_key_values[0].get_seq_length() == 3

    # Decode one token
    next_token = torch.tensor([[4]], dtype=torch.long)
    out2 = model(next_token, past_key_values=out1.past_key_values, use_cache=True)

    assert out2.past_key_values is not None
    assert out2.past_key_values[0].get_seq_length() == 4
    assert out2.logits.shape == (1, 1, config.vocab_size)


def test_generate_runs() -> None:
    config = _tiny_config()
    model = NumiForCausalLM(config)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = model.generate(input_ids, max_new_tokens=4, temperature=1.0)

    assert generated.shape[0] == 1
    assert generated.shape[1] == 7  # 3 + 4
