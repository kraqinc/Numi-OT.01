import torch

from architecture.attention.attention import Attention
from architecture.attention.cache import KVCache


def test_attention_forward_shape() -> None:
    attn = Attention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        max_position_embeddings=64,
    )
    x = torch.randn(2, 8, 64)
    out, cache = attn(x, use_cache=False)

    assert out.shape == (2, 8, 64)
    assert cache is None


def test_attention_with_cache() -> None:
    attn = Attention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        max_position_embeddings=64,
    )

    x1 = torch.randn(1, 4, 64)
    out1, cache = attn(x1, use_cache=True)
    assert cache is not None
    assert cache.get_seq_length() == 4
    assert out1.shape == (1, 4, 64)

    x2 = torch.randn(1, 1, 64)
    out2, cache2 = attn(x2, past_key_value=cache, use_cache=True, position_offset=4)
    assert cache2 is not None
    assert cache2.get_seq_length() == 5
    assert out2.shape == (1, 1, 64)


def test_kv_cache_update() -> None:
    cache = KVCache()
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    full_k, full_v = cache.update(k, v)
    assert full_k.shape[2] == 3
    assert cache.get_seq_length() == 3

    k2 = torch.randn(1, 2, 1, 8)
    v2 = torch.randn(1, 2, 1, 8)
    full_k2, _ = cache.update(k2, v2)
    assert full_k2.shape[2] == 4
