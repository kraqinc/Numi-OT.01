from __future__ import annotations

import torch
from torch import nn

from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM
from training.lora.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    lora_state_dict,
    mark_only_lora_as_trainable,
    merge_lora,
)


def test_lora_linear_shapes() -> None:
    base = nn.Linear(16, 32, bias=False)
    layer = LoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape == (2, 8, 32)


def test_apply_lora_on_numi() -> None:
    config = NumiConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = NumiForCausalLM(config)
    n = apply_lora(model, LoRAConfig(r=4, alpha=8.0, target_modules=("q_proj", "v_proj")))
    assert n > 0
    mark_only_lora_as_trainable(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)

    ids = torch.randint(0, 128, (1, 8))
    out = model(input_ids=ids, labels=ids)
    assert out.loss is not None
    out.loss.backward()

    sd = lora_state_dict(model)
    assert len(sd) > 0
    assert all("lora_" in k for k in sd)


def test_merge_lora() -> None:
    base = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.eye(8))
    layer = LoRALinear(base, r=2, alpha=2.0)
    with torch.no_grad():
        layer.lora_A.fill_(0.1)
        layer.lora_B.fill_(0.1)
    before = layer(torch.ones(1, 8)).clone()
    merge_lora(layer)
    # after merge, B is zeroed so delta is absorbed into base
    after = layer(torch.ones(1, 8))
    assert torch.allclose(before, after, atol=1e-5)
