from .token import LMHead, TokenEmbedding

# Aliases for backward compatibility with older imports
NumiTokenEmbedding = TokenEmbedding
NumiLMHead = LMHead

__all__ = [
    "TokenEmbedding",
    "LMHead",
    "NumiTokenEmbedding",
    "NumiLMHead",
]
