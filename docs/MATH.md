# Mathematics training data

| File | Role |
|------|------|
| `datasets/2026/numi_math_expert.jsonl` | Core school/college math explanations |
| `datasets/2026/numi_math_advanced.jsonl` | Algebra, calculus, probability, geometry, proofs |
| `datasets/2026/numi_math_code_extra.jsonl` | Short numeric drills |
| `datasets/2026/numi_reasoning.jsonl` | Multi-step logic |

Train:

```bash
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```
