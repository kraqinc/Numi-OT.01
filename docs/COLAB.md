# Colab — una sola celda

```python
import os
os.chdir("/content/Numi-OT.01")

!git pull
!pip install -q -r requirements-lora.txt
!pip install -q -e .

import sys
sys.path.insert(0, "/content/Numi-OT.01")

# Público (sin licencia):
!python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml

# O Gemma (solo si ya aceptaste https://huggingface.co/google/gemma-2-9b-it):
# !python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml --model-id google/gemma-2-9b-it

!python scripts/export_lora_safetensors.py

from serving.base_engine import BaseLoRAEngine
e = BaseLoRAEngine.from_adapter(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "outputs/lora_frontier/final",
)
print(e.generate([{"role":"user","content":"¿Qué modelo eres?"}]).text)
```

Artefactos:
- `outputs/lora_frontier/final/` (adapter PEFT completo)
- `outputs/lora_frontier/numi_lora.safetensors`
