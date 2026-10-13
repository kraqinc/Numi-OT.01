from architecture.attention.attention import Attention, repeat_kv
from architecture.blocks.mlp import NumiMLP
from architecture.blocks.transformer import TransformerBlock
from architecture.embeddings.token import LMHead, TokenEmbedding
from architecture.moe.moe import MoE
from architecture.normalization.rmsnorm import RMSNorm
from architecture.positional.rope import apply_rotary_emb, precompute_freqs_cis

# Public aliases (stable names used in docs / older imports)
NumiAttention = Attention
NumiTransformerBlock = TransformerBlock
NumiMoE = MoE
NumiRMSNorm = RMSNorm
NumiTokenEmbedding = TokenEmbedding
NumiLMHead = LMHead
build_rope_cache = precompute_freqs_cis
apply_rope = apply_rotary_emb

__all__ = [
    "Attention",
    "NumiAttention",
    "repeat_kv",
    "NumiMLP",
    "TransformerBlock",
    "NumiTransformerBlock",
    "TokenEmbedding",
    "NumiTokenEmbedding",
    "LMHead",
    "NumiLMHead",
    "MoE",
    "NumiMoE",
    "RMSNorm",
    "NumiRMSNorm",
    "precompute_freqs_cis",
    "build_rope_cache",
    "apply_rotary_emb",
    "apply_rope",
]
