from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re


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


class BPETokenizer:
    """
    Minimal reference BPE implementation.

    It is intentionally simple and deterministic. Production Numi
    tokenization can later use the trained tokenizer artifacts.
    """

    def __init__(
        self,
        vocabulary: BPEVocabulary,
        merges: list[tuple[str, str]] | None = None,
        unk_token: str = "<unk>",
    ) -> None:
        self.vocabulary = vocabulary
        self.merges = merges or []
        self.unk_token = unk_token

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        for word in self._pretokenize(text):
            symbols = list(word)

            for left, right in self.merges:
                merged = []
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
                ids.append(
                    self.vocabulary.token_to_id.get(
                        symbol,
                        self.vocabulary.token_to_id.get(
                            self.unk_token,
                            0,
                        ),
                    )
                )

        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.vocabulary.id_to_token.get(
                token_id,
                self.unk_token,
            )
            for token_id in ids
        )

    @staticmethod
    def learn_merges(
        texts: list[str],
        num_merges: int = 100,
    ) -> list[tuple[str, str]]:
        sequences = [
            list(text)
            for text in texts
            if text
        ]

        merges: list[tuple[str, str]] = []

        for _ in range(max(0, num_merges)):
            pairs = Counter()

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
