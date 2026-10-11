import torch

from architecture.attention.cache import KVCache


def test_cache_clear() -> None:
    cache = KVCache()
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    cache.update(k, v)
    assert cache.get_seq_length() == 4

    cache.clear()
    assert cache.get_seq_length() == 0
    assert cache.key is None
