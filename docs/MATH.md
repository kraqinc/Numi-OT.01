# Math track (Einstein-style habits, not magic)

Dataset: `datasets/2026/numi_math_einstein.jsonl`

Covers:
- Algebra, equations, logs, series
- Calculus (derivative, integral, limits, chain/product)
- Geometry / trig
- Probability, combinatorics
- Linear algebra basics
- Complex numbers, proofs (induction, contradiction)
- Fermi-style care with units / E=mc² intuition

Train:
```bash
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

Still not literal Einstein — but this is the right direction: steps, checks, clean final answer.
