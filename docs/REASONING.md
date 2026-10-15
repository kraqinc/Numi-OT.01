# Reasoning in Numi

## Product goal
Think when the problem needs it: math, bugs, architecture, logic.
Stay short on trivial facts.

## In code
- System prompt: step goal → steps → check → answer
- SFT: `datasets/2026/numi_reasoning.jsonl`
- RAG: `knowledge/learn` for new facts (not a substitute for reasoning)

## Train
```bash
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

## Honest limit
A 7B open model + LoRA will not become “the best AI ever created”.
It **can** be a strong independent assistant for Numination if you keep:
1) hard reasoning SFT, 2) live knowledge, 3) security, 4) GPU serving.
