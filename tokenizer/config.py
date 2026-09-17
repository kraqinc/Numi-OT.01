from dataclasses import dataclass


@dataclass(frozen=True)
class NumiTokenizerConfig:
    name: str = "Numi-OT.01"
    base_tokenizer: str = "Qwen/Qwen3-30B-A3B-Base"
    revision: str = "main"

    bos_token: str = "<|numi_bos|>"
    eos_token: str = "<|numi_eos|>"
    pad_token: str = "<|numi_pad|>"
    user_token: str = "<|numi_user|>"
    assistant_token: str = "<|numi_assistant|>"
    system_token: str = "<|numi_system|>"