from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .templates.chat import NumiChatTemplate, NumiMessage
from .config import NumiTokenizerConfig
from .special_tokens.tokens import NumiSpecialTokens


class NumiTokenizer:
    """
    High-level tokenizer wrapper for Numi-OT.01.

    The underlying tokenizer starts from the Qwen tokenizer and is extended
    with Numi-specific conversation tokens.
    """

    def __init__(
        self,
        config: NumiTokenizerConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config or NumiTokenizerConfig()
        self.special_tokens = NumiSpecialTokens()
        self._tokenizer = tokenizer

        self.chat_template = NumiChatTemplate.from_special_tokens(
            self.special_tokens
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | None = None,
        *,
        revision: str = "main",
        trust_remote_code: bool = False,
    ) -> NumiTokenizer:
        base_name = model_name or NumiTokenizerConfig().base_tokenizer

        config = NumiTokenizerConfig(
            base_tokenizer=base_name,
            revision=revision,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_tokenizer,
            revision=config.revision,
            trust_remote_code=trust_remote_code,
        )

        instance = cls(
            config=config,
            tokenizer=tokenizer,
        )

        instance.add_special_tokens()

        return instance

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        config: NumiTokenizerConfig | None = None,
    ) -> NumiTokenizer:
        instance = cls(
            config=config,
            tokenizer=tokenizer,
        )

        instance.add_special_tokens()

        return instance

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError(
                "NumiTokenizer is not initialized with a tokenizer. "
                "Use NumiTokenizer.from_pretrained() or from_tokenizer()."
            )

        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def base_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def added_tokens_count(self) -> int:
        return self.vocab_size - self.base_vocab_size

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        padding: bool | str = True,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ) -> Any:
        return self.tokenizer(
            list(texts),
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

    def add_special_tokens(self) -> int:
        """
        Add the complete Numi vocabulary extension.

        Returns:
            Number of newly added tokens.
        """
        tokens = self.special_tokens

        special_tokens = {
            "bos_token": tokens.bos,
            "eos_token": tokens.eos,
            "pad_token": tokens.pad,
            "additional_special_tokens": [
                tokens.system,
                tokens.user,
                tokens.assistant,
                tokens.tool,
                tokens.tool_result,
            ],
        }

        return self.tokenizer.add_special_tokens(special_tokens)

    def token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            raise KeyError(f"Token '{token}' is not present in the tokenizer.")

        if token_id == self.tokenizer.unk_token_id and token != self.tokenizer.unk_token:
            raise KeyError(f"Token '{token}' is not present in the tokenizer.")

        return int(token_id)

    def special_token_ids(self) -> dict[str, int]:
        tokens = self.special_tokens

        return {
            "bos": self.token_id(tokens.bos),
            "eos": self.token_id(tokens.eos),
            "pad": self.token_id(tokens.pad),
            "system": self.token_id(tokens.system),
            "user": self.token_id(tokens.user),
            "assistant": self.token_id(tokens.assistant),
            "tool": self.token_id(tokens.tool),
            "tool_result": self.token_id(tokens.tool_result),
        }

    def render_chat(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        return self.chat_template.render(
            messages,
            add_generation_prompt=add_generation_prompt,
        )

    def encode_chat(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
        *,
        add_generation_prompt: bool = True,
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> list[int]:
        prompt = self.render_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
        )

        return self.encode(
            prompt,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

    def tokenize_chat(
        self,
        messages: Iterable[Mapping[str, object] | NumiMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> list[str]:
        prompt = self.render_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
        )

        return self.tokenizer.tokenize(prompt)

    def save_pretrained(self, save_directory: str | Path) -> tuple[str, ...]:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        return self.tokenizer.save_pretrained(path)