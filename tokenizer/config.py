from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NumiTokenizerConfig:
    """
    Configuration for the Numi tokenizer.

    Completely independent. No external model is required as a hard dependency.
    A base tokenizer can be provided later for training or conversion,
    but Numi owns its own special tokens and chat format.
    """

    name: str = "Numi-OT.01"
    version: str = "0.1.0"

    # Optional base (can be any HF tokenizer or a trained BPE later)
    # Set to None when using a fully trained Numi tokenizer.
    base_tokenizer: str | None = None
    revision: str = "main"

    # Core special tokens
    bos_token: str = "<|numi_bos|>"
    eos_token: str = "<|numi_eos|>"
    pad_token: str = "<|numi_pad|>"
    unk_token: str = "<|numi_unk|>"

    # Conversation roles
    system_token: str = "<|numi_system|>"
    user_token: str = "<|numi_user|>"
    assistant_token: str = "<|numi_assistant|>"

    # Tool use
    tool_token: str = "<|numi_tool|>"
    tool_result_token: str = "<|numi_tool_result|>"

    # Extra control tokens
    start_header: str = "<|numi_start|>"
    end_header: str = "<|numi_end|>"

    model_max_length: int = 8192

    # Training / vocabulary
    target_vocab_size: int = 128256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def all_special_tokens(self) -> list[str]:
        return [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.unk_token,
            self.system_token,
            self.user_token,
            self.assistant_token,
            self.tool_token,
            self.tool_result_token,
            self.start_header,
            self.end_header,
        ]
