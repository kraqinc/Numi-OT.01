from architecture.positional.rope import apply_rotary_emb, precompute_freqs_cis

# Aliases for older imports
build_rope_cache = precompute_freqs_cis
apply_rope = apply_rotary_emb

__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "build_rope_cache",
    "apply_rope",
]
