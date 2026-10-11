from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from model.configs.numi_config import NumiConfig
from model.modeling.numi_causal_lm import NumiForCausalLM
from tokenizer.special_tokens.tokens import NumiSpecialTokens
from tokenizer.templates.chat import NumiChatTemplate
from training.sft.train import SimpleTokenizer, get_device


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str


class PreferenceDataset(Dataset):
    """JSONL with fields: prompt, chosen, rejected (or messages + chosen/rejected)."""

    def __init__(self, path: str | Path) -> None:
        self.examples: list[PreferenceExample] = []
        special = NumiSpecialTokens()
        template = NumiChatTemplate.from_special_tokens(special)

        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "prompt" in row:
                    prompt = row["prompt"]
                elif "messages" in row:
                    prompt = template.render(row["messages"], add_generation_prompt=True)
                else:
                    continue
                self.examples.append(
                    PreferenceExample(
                        prompt=prompt,
                        chosen=str(row["chosen"]),
                        rejected=str(row["rejected"]),
                    )
                )

        if not self.examples:
            raise ValueError(f"No preference examples in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreferenceExample:
        return self.examples[index]


def logps_for_completion(
    model: NumiForCausalLM,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    """Average log-prob of tokens after the prompt."""
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [1, S, V]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_logps = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    # only completion tokens (positions >= prompt_len-1 in shifted space)
    start = max(prompt_len - 1, 0)
    completion_logps = token_logps[:, start:]
    return completion_logps.sum(dim=-1)  # [1]


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    ref_chosen: Tensor,
    ref_rejected: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Standard DPO loss (Rafailov et al.)."""
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = beta * (pi_logratios - ref_logratios)
    return -torch.nn.functional.logsigmoid(logits).mean()


def train_dpo(config: dict[str, Any]) -> Path:
    device = get_device(config["training"].get("device", "auto"))
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SimpleTokenizer(vocab_size=config["model"]["vocab_size"])
    max_len = config["data"]["max_seq_length"]

    policy = NumiForCausalLM(NumiConfig(**config["model"])).to(device)
    ref = NumiForCausalLM(NumiConfig(**config["model"])).to(device)

    # Optional: load SFT checkpoint into both
    ckpt_path = config["training"].get("sft_checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(state["model"])
        ref.load_state_dict(state["model"])

    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    dataset = PreferenceDataset(config["data"]["path"])
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    beta = float(config["training"].get("beta", 0.1))
    policy.train()

    step = 0
    for epoch in range(config["training"]["num_epochs"]):
        for example in loader:
            # DataLoader with batch_size=1 yields batched strings oddly; handle both
            if isinstance(example, dict):
                prompt = example["prompt"][0] if isinstance(example["prompt"], list) else example.prompt
                chosen = example["chosen"][0] if isinstance(example["chosen"], list) else example.chosen
                rejected = example["rejected"][0] if isinstance(example["rejected"], list) else example.rejected
            else:
                # default collate on dataclass becomes lists
                prompt = example.prompt[0] if isinstance(example.prompt, (list, tuple)) else example.prompt
                chosen = example.chosen[0] if isinstance(example.chosen, (list, tuple)) else example.chosen
                rejected = example.rejected[0] if isinstance(example.rejected, (list, tuple)) else example.rejected

            # Prefer PreferenceExample attributes when collate keeps objects
            if hasattr(example, "prompt") and not isinstance(example, dict):
                # pytorch default collate converts dataclass to dict-like via fields
                pass

            # Robust path: dataset returns PreferenceExample; default_collate fails on dataclass
            # so we use batch_size=1 with a custom approach below.

    # Custom loop without default_collate issues
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config["training"]["num_epochs"]):
        for i in range(len(dataset)):
            ex = dataset[i]
            prompt_ids = tokenizer.encode(ex.prompt, max_length=max_len // 2)
            chosen_ids = tokenizer.encode(ex.prompt + ex.chosen, max_length=max_len)
            rejected_ids = tokenizer.encode(ex.prompt + ex.rejected, max_length=max_len)

            chosen_tensor = torch.tensor([chosen_ids], dtype=torch.long, device=device)
            rejected_tensor = torch.tensor([rejected_ids], dtype=torch.long, device=device)
            prompt_len = len(prompt_ids)

            policy_chosen = logps_for_completion(policy, chosen_tensor, prompt_len)
            policy_rejected = logps_for_completion(policy, rejected_tensor, prompt_len)

            with torch.no_grad():
                ref_chosen = logps_for_completion(ref, chosen_tensor, prompt_len)
                ref_rejected = logps_for_completion(ref, rejected_tensor, prompt_len)

            loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config["training"].get("max_grad_norm", 1.0))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % config["training"].get("logging_steps", 5) == 0:
                print(f"epoch={epoch} step={step} dpo_loss={loss.item():.4f}")

    final_path = output_dir / "dpo_final.pt"
    torch.save({"model": policy.state_dict(), "config": config, "step": step}, final_path)
    print(f"DPO complete → {final_path}")
    return final_path
