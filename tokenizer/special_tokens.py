from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NumiSpecialTokens:
    bos: str = "<|numi_bos|>"
    eos: str = "<|numi_eos|>"
    pad: str = "<|numi_pad|>"

    system: str = "<|numi_system|>"
    user: str = "<|numi_user|>"
    assistant: str = "<|numi_assistant|>"

    tool: str = "<|numi_tool|>"
    tool_result: str = "<|numi_tool_result|>"

    def all(self) -> tuple[str, ...]:
        return (
            self.bos,
            self.eos,
            self.pad,
            self.system,
            self.user,
            self.assistant,
            self.tool,
            self.tool_result,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)