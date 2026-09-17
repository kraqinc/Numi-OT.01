from dataclasses import dataclass


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