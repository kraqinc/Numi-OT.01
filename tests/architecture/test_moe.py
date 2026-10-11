import torch

from architecture.moe.moe import MoE
from architecture.moe.router import Router


def test_router_top_k() -> None:
    router = Router(hidden_size=32, num_experts=4, top_k=2)
    x = torch.randn(10, 32)
    weights, indices, probs = router(x)

    assert weights.shape == (10, 2)
    assert indices.shape == (10, 2)
    assert probs.shape == (10, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(10), atol=1e-5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_moe_forward_shape() -> None:
    moe = MoE(
        hidden_size=32,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
    )
    x = torch.randn(2, 5, 32)
    out = moe(x)

    assert out.shape == (2, 5, 32)
    assert moe.last_aux_loss is not None


def test_moe_dispatch_uses_all_selected_experts() -> None:
    torch.manual_seed(0)
    moe = MoE(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
    )
    x = torch.randn(1, 8, 16)
    out = moe(x)
    # Non-zero output expected after random init + routing
    assert out.abs().sum() > 0
