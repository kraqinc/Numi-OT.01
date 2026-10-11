from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class BPEVocabulary:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "BPEVocabulary":
        token_to_id = {token: index for index, token in enumerate(tokens)}
        return cls(
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
        )

    def __len__(self) -> int:
        return len(self.token_to_id)


@dataclass
class BPETokenizer:
    """
    Trainable BPE tokenizer for Numi.

    Special tokens are reserved at the start of the vocabulary.
    """

    vocabulary: BPEVocabulary
    merges: list[tuple[str, str]] = field(default_factory=list)
    unk_token: str = "<|numi_unk|>"
    special_tokens: list[str] = field(default_factory=list)

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        ids: list[int] = []
        unk_id = self.vocabulary.token_to_id.get(self.unk_token, 0)

        # Preserve multi-char special tokens if present as whole words
        for word in self._pretokenize(text):
            if word in self.vocabulary.token_to_id:
                ids.append(self.vocabulary.token_to_id[word])
                if max_length and len(ids) >= max_length:
                    return ids[:max_length]
                continue

            symbols = list(word)
            for left, right in self.merges:
                merged: list[str] = []
                index = 0
                while index < len(symbols):
                    if (
                        index + 1 < len(symbols)
                        and symbols[index] == left
                        and symbols[index + 1] == right
                    ):
                        merged.append(left + right)
                        index += 2
                    else:
                        merged.append(symbols[index])
                        index += 1
                symbols = merged

            for symbol in symbols:
                ids.append(self.vocabulary.token_to_id.get(symbol, unk_id))
                if max_length and len(ids) >= max_length:
                    return ids[:max_length]

        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        special = set(self.special_tokens)
        pieces: list[str] = []
        for token_id in ids:
            token = self.vocabulary.id_to_token.get(int(token_id), self.unk_token)
            if skip_special_tokens and token in special:
                continue
            pieces.append(token)
        return "".join(pieces)

    @staticmethod
    def learn_merges(texts: list[str], num_merges: int = 1000) -> list[tuple[str, str]]:
        sequences = [list(text) for text in texts if text]
        merges: list[tuple[str, str]] = []

        for _ in range(max(0, num_merges)):
            pairs: Counter[tuple[str, str]] = Counter()
            for sequence in sequences:
                pairs.update(zip(sequence, sequence[1:]))
            if not pairs:
                break

            pair, _ = pairs.most_common(1)[0]
            merges.append(pair)
            left, right = pair

            for index, sequence in enumerate(sequences):
                merged: list[str] = []
                cursor = 0
                while cursor < len(sequence):
                    if (
                        cursor + 1 < len(sequence)
                        and sequence[cursor] == left
                        and sequence[cursor + 1] == right
                    ):
                        merged.append(left + right)
                        cursor += 2
                    else:
                        merged.append(sequence[cursor])
                        cursor += 1
                sequences[index] = merged

        return merges

    @classmethod
    def train(
        cls,
        texts: list[str],
        *,
        vocab_size: int = 8000,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        special_tokens = special_tokens or []
        # Base alphabet from characters present in data + specials
        alphabet = sorted({ch for text in texts for ch in text})
        base_tokens = list(special_tokens)
        for ch in alphabet:
            if ch not in base_tokens:
                base_tokens.append(ch)

        # Number of merges limited by remaining vocab budget
        num_merges = max(0, vocab_size - len(base_tokens))
        merges = cls.learn_merges(texts, num_merges=num_merges)

        # Build final vocab: specials + alphabet + merge products in order
        tokens = list(base_tokens)
        seen = set(tokens)
        for left, right in merges:
            piece = left + right
            if piece not in seen:
                tokens.append(piece)
                seen.add(piece)
            if len(tokens) >= vocab_size:
                break

        vocab = BPEVocabulary.from_tokens(tokens[:vocab_size])
        return cls(
            vocabulary=vocab,
            merges=merges,
            unk_token=special_tokens[3] if len(special_tokens) > 3 else "<|numi_unk|>",
            special_tokens=list(special_tokens),
        )

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "unk_token": self.unk_token,
            "special_tokens": self.special_tokens,
            "merges": [list(m) for m in self.merges],
            "tokens": [
                self.vocabulary.id_to_token[i]
                for i in range(len(self.vocabulary))
            ],
        }
        (path / "tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        path = Path(directory) / "tokenizer.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = BPEVocabulary.from_tokens(payload["tokens"])
        merges = [tuple(m) for m in payload["merges"]]
        return cls(
            vocabulary=vocab,
            merges=merges,
            unk_token=payload.get("unk_token", "<|numi_unk|>"),
            special_tokens=payload.get("special_tokens", []),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)
