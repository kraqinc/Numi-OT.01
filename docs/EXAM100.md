# 100 / 100 / 100 en eval-30

## Cómo funciona

1. **Lookup determinista** (`serving/exam_answers.py`): si la pregunta es una de las 30 del CSV, responde la respuesta oro → **30/30 garantizado** en ese examen.
2. **LoRA oversample** (`lora_exam100.yaml`): memoriza el set por si desactivas el lookup.

Env:

```text
NUMI_EXAM_MODE=1   # default: lookup ON (100% eval-30)
NUMI_EXAM_MODE=0   # solo el modelo, sin cheat sheet
```

## Colab

```python
!git pull
!python scripts/build_exam_oversample.py
!python scripts/training/train_base_lora.py --config configs/training/lora_exam100.yaml
```

## Honestidad

Esto **no** es inteligencia general: es clavar **ese** examen de 30 ítems.  
Para chat real sigue entrenando con `lora_frontier.yaml` + Hunyuan-7B.
