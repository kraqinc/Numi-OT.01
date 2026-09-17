from collections.abc import Sequence
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import NumiTokenizerConfig
from .special_tokens import NumiSpecialTokens


class NumiTokenizer:
    def __init__(
        self,
        config: NumiTokenizerConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config or NumiTokenizerConfig()
        self.special_tokens = NumiSpecialTokens()

        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | None = None,
        revision: str = "main",
    ) -> "NumiTokenizer":
        config = NumiTokenizerConfig(
            base_tokenizer=model_name
            or NumiTokenizerConfig.base_tokenizer,
            revision=revision,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_tokenizer,
            revision=config.revision,
        )

        return cls(
            config=config,
            tokenizer=tokenizer,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer has not been loaded. "
                "Use NumiTokenizer.from_pretrained()."
            )

        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        padding: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        return self.tokenizer(
            list(texts),
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

    def add_special_tokens(self) -> int:
        return self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self.special_tokens.bos,
                    self.special_tokens.eos,
                    self.special_tokens.pad,
                    self.special_tokens.system,
                    self.special_tokens.user,
                    self.special_tokens.assistant,
                    self.special_tokens.tool,
                    self.special_tokens.tool_result,
                ]
            }
        )