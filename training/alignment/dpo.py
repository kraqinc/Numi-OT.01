from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

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


class PreferenceDataset:
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
    """Sum of log-probs of tokens after the prompt."""
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_logps = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    start = max(prompt_len - 1, 0)
    return token_logps[:, start:].sum(dim=-1)


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
    max_len = int(config["data"]["max_seq_length"])

    policy = NumiForCausalLM(NumiConfig(**config["model"])).to(device)
    ref = NumiForCausalLM(NumiConfig(**config["model"])).to(device)

    ckpt_path = config["training"].get("sft_checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(state["model"])
        ref.load_state_dict(state["model"])

    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    dataset = PreferenceDataset(config["data"]["path"])
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    beta = float(config["training"].get("beta", 0.1))
    policy.train()
    step = 0

    print(f"Device: {device}")
    print(f"Preference pairs: {len(dataset)}")

    for epoch in range(int(config["training"]["num_epochs"])):
        for i in range(len(dataset)):
            ex = dataset[i]
            prompt_ids = tokenizer.encode(ex.prompt, max_length=max_len // 2)
            chosen_ids = tokenizer.encode(ex.prompt + ex.chosen, max_length=max_len)
            rejected_ids = tokenizer.encode(ex.prompt + ex.rejected, max_length=max_len)

            if len(chosen_ids) < 2 or len(rejected_ids) < 2:
                continue

            chosen_tensor = torch.tensor([chosen_ids], dtype=torch.long, device=device)
            rejected_tensor = torch.tensor([rejected_ids], dtype=torch.long, device=device)
            prompt_len = len(prompt_ids)

            policy_chosen = logps_for_completion(policy, chosen_tensor, prompt_len)
            policy_rejected = logps_for_completion(policy, rejected_tensor, prompt_len)

            with torch.no_grad():
                ref_chosen = logps_for_completion(ref, chosen_tensor, prompt_len)
                ref_rejected = logps_for_completion(ref, rejected_tensor, prompt_len)

            loss = dpo_loss(
                policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                policy.parameters(), config["training"].get("max_grad_norm", 1.0)
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % int(config["training"].get("logging_steps", 5)) == 0:
                print(f"epoch={epoch} step={step} dpo_loss={loss.item():.4f}")

    final_path = output_dir / "dpo_final.pt"
    torch.save({"model": policy.state_dict(), "config": config, "step": step}, final_path)
    print(f"DPO complete → {final_path}")
    return final_path
