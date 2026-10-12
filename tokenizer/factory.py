from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from tokenizer.algorithms.bpe import BPETokenizer
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from tokenizer.templates.chat import escape_control_tokens


class TokenizerProtocol(Protocol):
    vocab_size: int
    pad_id: int

    def encode(self, text: str, max_length: int | None = None) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...

    def pad_batch(
        self, sequences: list[list[int]], max_length: int
    ) -> tuple[Tensor, Tensor]: ...


class SimpleTokenizer:
    """Byte-level fallback. Special tokens are NOT matched inside user content
    when encode_content=True (default for untrusted text).
    """

    def __init__(self, vocab_size: int = 128256) -> None:
        self.special = NumiSpecialTokens()
        self.special_list = list(self.special.all())
        self.vocab_size = max(vocab_size, 256 + len(self.special_list) + 8)
        self.pad_id = 0
        self.unk_id = 1
        self.special_to_id = {tok: 2 + i for i, tok in enumerate(self.special_list)}
        self.byte_offset = 2 + len(self.special_list)

    def encode(
        self,
        text: str,
        max_length: int | None = None,
        *,
        allow_special: bool = True,
    ) -> list[int]:
        """Encode text to ids.

        allow_special=False: escape control markers first (safe for user content).
        allow_special=True: match specials (only for fully trusted template strings).
        """
        if not allow_special:
            text = escape_control_tokens(text)
            # After escape, still do not match specials — pure byte path
            allow_special = False

        limit = max_length or 10_000_000
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            if allow_special:
                for tok, tid in self.special_to_id.items():
                    if text.startswith(tok, i):
                        ids.append(tid)
                        i += len(tok)
                        matched = True
                        break
            if matched:
                if len(ids) >= limit:
                    break
                continue
            for byte in text[i].encode("utf-8"):
                ids.append(self.byte_offset + (byte % 256))
                if len(ids) >= limit:
                    return ids[:limit]
            i += 1
        return ids[:limit]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        id_to_special = {v: k for k, v in self.special_to_id.items()}
        pieces: list[str] = []
        for tid in ids:
            if tid in id_to_special:
                if skip_special_tokens:
                    continue
                pieces.append(id_to_special[tid])
            elif tid >= self.byte_offset:
                byte = (tid - self.byte_offset) % 256
                pieces.append(bytes([byte]).decode("utf-8", errors="ignore"))
        return "".join(pieces)

    def pad_batch(
        self, sequences: list[list[int]], max_length: int
    ) -> tuple[Tensor, Tensor]:
        batch, labels = [], []
        for seq in sequences:
            padded = seq + [self.pad_id] * (max_length - len(seq))
            label = seq + [-100] * (max_length - len(seq))
            batch.append(padded[:max_length])
            labels.append(label[:max_length])
        return torch.tensor(batch, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class BPETokenizerAdapter:
    def __init__(self, tokenizer: BPETokenizer) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        pad_token = NumiSpecialTokens().pad
        self.pad_id = tokenizer.vocabulary.token_to_id.get(pad_token, 0)
        self.unk_id = tokenizer.vocabulary.token_to_id.get(
            tokenizer.unk_token, self.pad_id
        )

    def encode(
        self,
        text: str,
        max_length: int | None = None,
        *,
        allow_special: bool = True,
    ) -> list[int]:
        if not allow_special:
            text = escape_control_tokens(text)
        return self.tokenizer.encode(text, max_length=max_length)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def pad_batch(
        self, sequences: list[list[int]], max_length: int
    ) -> tuple[Tensor, Tensor]:
        batch, labels = [], []
        for seq in sequences:
            padded = seq + [self.pad_id] * (max_length - len(seq))
            label = seq + [-100] * (max_length - len(seq))
            batch.append(padded[:max_length])
            labels.append(label[:max_length])
        return torch.tensor(batch, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def load_tokenizer(
    tokenizer_path: str | Path | None = None,
    vocab_size: int = 128256,
) -> SimpleTokenizer | BPETokenizerAdapter:
    if tokenizer_path:
        path = Path(tokenizer_path)
        if (path / "tokenizer.json").exists():
            return BPETokenizerAdapter(BPETokenizer.load(path))
        if path.exists() and path.is_file() and path.name == "tokenizer.json":
            return BPETokenizerAdapter(BPETokenizer.load(path.parent))

    default = Path("artifacts/tokenizer")
    if (default / "tokenizer.json").exists():
        return BPETokenizerAdapter(BPETokenizer.load(default))

    return SimpleTokenizer(vocab_size=vocab_size)
