def test_architecture_modules_import():
    from architecture.modules import (
        Attention,
        LMHead,
        MoE,
        NumiAttention,
        NumiLMHead,
        NumiMoE,
        NumiRMSNorm,
        NumiTransformerBlock,
        RMSNorm,
        TokenEmbedding,
        TransformerBlock,
        apply_rope,
        build_rope_cache,
        repeat_kv,
    )

    assert Attention is NumiAttention
    assert TransformerBlock is NumiTransformerBlock
    assert MoE is NumiMoE
    assert RMSNorm is NumiRMSNorm
    assert LMHead is NumiLMHead
    assert callable(repeat_kv)
    assert callable(build_rope_cache)
    assert callable(apply_rope)
    assert TokenEmbedding is not None
