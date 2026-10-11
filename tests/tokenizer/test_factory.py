from tokenizer.factory import BPETokenizerAdapter, SimpleTokenizer, load_tokenizer
from tokenizer.algorithms.bpe import BPETokenizer
from tokenizer.special_tokens.tokens import NumiSpecialTokens


def test_simple_tokenizer_pad_batch() -> None:
    tok = SimpleTokenizer(vocab_size=512)
    ids = tok.encode("Hola Numi", max_length=32)
    batch, labels = tok.pad_batch([ids, ids[:3]], max_length=16)
    assert batch.shape == (2, 16)
    assert labels.shape == (2, 16)


def test_load_tokenizer_falls_back() -> None:
    tok = load_tokenizer(tokenizer_path="does/not/exist", vocab_size=1024)
    assert isinstance(tok, SimpleTokenizer)


def test_bpe_adapter(tmp_path) -> None:
    special = list(NumiSpecialTokens().all())
    bpe = BPETokenizer.train(["Hola Numi mundo"], vocab_size=100, special_tokens=special)
    bpe.save(tmp_path)
    adapter = load_tokenizer(tmp_path, vocab_size=100)
    assert isinstance(adapter, BPETokenizerAdapter)
    ids = adapter.encode("Hola", max_length=32)
    assert len(ids) > 0
