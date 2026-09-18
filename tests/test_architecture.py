import torch

from architecture import NumiConfig, NumiForCausalLM, NumiModel
from architecture.modules import NumiTokenEmbedding


def test_configuration_defaults() -> None:
    config = NumiConfig()

    assert config.vocab_size == 151936
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 48
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 4
    assert config.num_experts == 128
    assert config.num_experts_per_tok == 8


def test_embedding_shape() -> None:
    embedding = NumiTokenEmbedding(
        vocab_size=100,
        hidden_size=32,
    )

    input_ids = torch.tensor([[1, 2, 3]])

    output = embedding(input_ids)

    assert output.shape == (1, 3, 32)


def test_embedding_resize_preserves_old_weights() -> None:
    embedding = NumiTokenEmbedding(
        vocab_size=10,
        hidden_size=8,
    )

    original = embedding.weight.detach().clone()

    embedding.resize(13)

    assert embedding.vocab_size == 13
    assert embedding.weight.shape == (13, 8)
    assert torch.equal(
        embedding.weight[:10],
        original,
    )


def test_model_resize() -> None:
    config = NumiConfig(
        vocab_size=100,
        hidden_size=32,
    )

    model = NumiModel(config)

    original = model.embeddings.weight.detach().clone()

    model.resize_token_embeddings(108)

    assert model.config.vocab_size == 108
    assert model.embeddings.weight.shape == (108, 32)

    assert torch.equal(
        model.embeddings.weight[:100],
        original,
    )


def test_causal_lm_forward() -> None:
    config = NumiConfig(
        vocab_size=100,
        hidden_size=32,
    )

    model = NumiForCausalLM(config)

    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    output = model(
        input_ids=input_ids,
        labels=input_ids,
    )

    assert output.logits is not None
    assert output.loss is not None
    assert output.logits.shape == (1, 4, 100)


def test_tied_embeddings() -> None:
    config = NumiConfig(
        vocab_size=100,
        hidden_size=32,
        tie_word_embeddings=True,
    )

    model = NumiForCausalLM(config)

    assert (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    )


def test_causal_lm_resize_keeps_tying() -> None:
    config = NumiConfig(
        vocab_size=100,
        hidden_size=32,
        tie_word_embeddings=True,
    )

    model = NumiForCausalLM(config)

    model.resize_token_embeddings(108)

    assert model.config.vocab_size == 108
    assert model.get_output_embeddings().shape == (108, 32)

    assert (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    )