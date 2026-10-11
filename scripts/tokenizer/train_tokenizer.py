#!/usr/bin/env python3
"""Train a Numi BPE tokenizer from conversation / text data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizer.algorithms.bpe import BPETokenizer
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from tokenizer.templates.chat import NumiChatTemplate


def load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    special = NumiSpecialTokens()
    template = NumiChatTemplate.from_special_tokens(special)

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "messages" in record:
                    texts.append(
                        template.render(record["messages"], add_generation_prompt=False)
                    )
                elif "text" in record:
                    texts.append(str(record["text"]))
    else:
        texts.append(path.read_text(encoding="utf-8"))

    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Numi BPE tokenizer")
    parser.add_argument(
        "--data",
        default="datasets/2026/numi_conversation_seed.jsonl",
        help="JSONL or text file",
    )
    parser.add_argument("--output", default="artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=4000)
    args = parser.parse_args()

    texts = load_texts(Path(args.data))
    if not texts:
        raise SystemExit(f"No texts found in {args.data}")

    special = NumiSpecialTokens()
    tokenizer = BPETokenizer.train(
        texts,
        vocab_size=args.vocab_size,
        special_tokens=list(special.all()),
    )
    tokenizer.save(args.output)

    sample = texts[0][:120]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print(f"Saved tokenizer → {args.output}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Merges: {len(tokenizer.merges)}")
    print(f"Sample encode length: {len(encoded)}")
    print(f"Decode starts with: {decoded[:80]!r}")


if __name__ == "__main__":
    main()
