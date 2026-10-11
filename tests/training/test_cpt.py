from pathlib import Path

from training.cpt.train import stream_text_lines


def test_stream_conversation_jsonl() -> None:
    path = Path("datasets/2026/numi_conversation_seed.jsonl")
    lines = list(stream_text_lines(path))
    assert len(lines) >= 5
    assert any(len(x) > 0 for x in lines)
