import torch

from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM


def test_generate_respects_max_new_tokens() -> None:
    config = NumiConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=64,
    )
    model = NumiForCausalLM(config)
    model.eval()

    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=3)
    assert out.shape == (1, 5)
