# Deprecated: exam cheat mode

**No uses** el lookup de examen para producto real.

Usuarios no piden un examen de 30 ítems: necesitan mates, código, español, geografía y cultura de verdad.

Entrena con:

```bash
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

Datasets clave:
- `numi_math_expert.jsonl`
- `numi_code_expert.jsonl`
- `numi_spanish_expert.jsonl`
- `numi_geo_culture.jsonl`

`serving/exam_answers.py` puede existir por legado; el engine **ya no** lo usa por defecto.
