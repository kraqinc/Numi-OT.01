# Reasoning

- System prompt: goal → steps → check → answer
- Data: `datasets/2026/numi_reasoning.jsonl`
- Live facts: `knowledge/learn/` + RAG (`NUMI_RAG=1`)

```bash
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```
