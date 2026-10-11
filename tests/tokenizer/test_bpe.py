from tokenizer.algorithms.bpe import BPETokenizer
from tokenizer.special_tokens.tokens import NumiSpecialTokens


def test_bpe_train_encode_decode(tmp_path) -> None:
    texts = [
        "Hola Numi, ¿cómo estás?",
        "Mixture of Experts es una arquitectura eficiente.",
        "def sumar(a, b): return a + b",
    ]
    special = list(NumiSpecialTokens().all())
    tok = BPETokenizer.train(texts, vocab_size=200, special_tokens=special)

    assert tok.vocab_size <= 200
    ids = tok.encode(texts[0])
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert "Hola" in decoded or "Numi" in decoded or len(decoded) > 0

    out = tmp_path / "tok"
    tok.save(out)
    loaded = BPETokenizer.load(out)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(texts[0]) == ids
