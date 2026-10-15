# Eval-30 (Numi vs baseline)

Del CSV real:

| Modelo | Aprox. aciertos |
|--------|------------------|
| **Numi-v0.2** | ~10/30 — **muy débil en mates/código** |
| Qwen2.5-1.5B | ~12/30 — un poco mejor en mates, peor siguiendo formato |

Fallos típicos Numi: `17*23→489`, `15% de 200→15`, `2**10→20`, `DISTINCT→DELETE`.

## Fix en código (este repo)

1. `datasets/2026/numi_eval30_sft.jsonl` — las 30 Q con respuesta exacta  
2. `datasets/2026/numi_math_code_extra.jsonl` — más mates/Python cortos  
3. System prompt: disciplina de formato (“solo el número”)  
4. `lora_frontier.yaml` incluye esos JSONL + `temperature` infer 0.2  

## Re-entrenar

```bash
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

Luego re-correr el bench. Objetivo: ≥25/30 en eval-30 antes de llamar “v0.3”.

Nota: memorizar el set no = inteligencia general; sirve como regresión mínima.
