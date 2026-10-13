# Más SFT + base más grande

## Datos

Además de los seeds, usa `datasets/2026/numi_sft_quality.jsonl` (respuestas cortas, sin bucles, producto Numi, código, deploy).

## Bases (sin Qwen)

| Config | Base | VRAM aprox 4-bit |
|--------|------|------------------|
| `lora_frontier.yaml` | **Phi-3.5-mini-instruct** (~3.8B) | Colab T4 16GB OK |
| `lora_large.yaml` | **Hunyuan-7B-Instruct** (~7B) | L4/A100 / RunPod |

Si Phi-3.5 pide licencia en HF: abre el model card, Accept, con la misma cuenta del `huggingface-cli login`.

## Colab

```python
%cd /content/Numi-OT.01
!git pull
!pip install -q -r requirements-lora.txt
!pip install -q -e .

# 3.8B (recomendado primero)
!python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml

# o 7B si tienes GPU grande:
# !python scripts/training/train_base_lora.py --config configs/training/lora_large.yaml

!python scripts/export_lora_safetensors.py --adapter outputs/lora_frontier/final
```

Prueba:

```python
from serving.base_engine import BaseLoRAEngine
e = BaseLoRAEngine.from_adapter(
    "microsoft/Phi-3.5-mini-instruct",
    "outputs/lora_frontier/final",
    load_in_4bit=True,
)
print(e.generate([{"role":"user","content":"¿Me ayudas?"}], max_tokens=128).text)
```

Luego sube el adapter a HF y apunta Render `NUMI_PEFT_BASE` + `NUMI_PEFT_ADAPTER`.
