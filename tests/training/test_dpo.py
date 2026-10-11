from pathlib import Path

import torch

from training.alignment.dpo import PreferenceDataset, dpo_loss


def test_preference_dataset_loads() -> None:
    path = Path("datasets/2026/numi_preferences_seed.jsonl")
    assert path.exists()
    ds = PreferenceDataset(path)
    assert len(ds) >= 3
    assert ds[0].chosen != ds[0].rejected


def test_dpo_loss_runs() -> None:
    policy_c = torch.tensor([1.0])
    policy_r = torch.tensor([0.2])
    ref_c = torch.tensor([0.9])
    ref_r = torch.tensor([0.3])
    loss = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
