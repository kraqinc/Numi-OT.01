# Numi real: LoRA sobre base open (octubre 2026)

## Ranking open-weight (referencia)

| Nivel | Modelos | Hardware |
|-------|---------|----------|
| Frontier (datacenter) | **MiMo-V2.6-Pro**, Kimi K3, GLM-5.3 | multi-GPU 80GB |
| Fuerte + 1 GPU | **gpt-oss-20b**, DeepSeek-V4-Flash, Gemma-4 12B/31B | 16–24GB+ 4-bit |
| Evitado por producto | Qwen* | — (regla Numination) |

Default del repo: **`openai/gpt-oss-20b`** (Apache 2.0, fuerte, fine-tuneable).
Cambia `base.model_id` en `configs/training/lora_frontier.yaml` si tienes VRAM para MiMo/Kimi/DeepSeek.

## Colab / GPU

```python
from getpass import getpass
token = getpass("HF+GitHub as needed: ")

!git clone https://{token}@github.com/kraqinc/Numi-OT.01.git
%cd Numi-OT.01
!pip install -q -r requirements-dev.txt peft bitsandbytes
!pip install -q -e .

# Runtime → GPU
!python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml

# O override del base:
# !python scripts/training/train_base_lora.py --model-id deepseek-ai/DeepSeek-V4-Flash-0731
```

Adapter out: `outputs/lora_frontier/final/`

## Inferencia

```python
from serving.base_engine import BaseLoRAEngine

e = BaseLoRAEngine.from_adapter(
    base_model_id="openai/gpt-oss-20b",
    adapter_path="outputs/lora_frontier/final",
    load_in_4bit=True,
)
print(e.generate([{"role":"user","content":"¿Qué modelo eres?"}]).text)
```

## Producto

- System prompt **siempre** Numi OT (forzado en data + engine).
- API FastAPI puede envolver `BaseLoRAEngine` igual que el toy engine.
- Supabase sin cambios de arquitectura.

## Honestidad

Esto **sí** puede ser un chat de producto. El toy MoE Numi desde cero no.
El adapter LoRA es Numi por identidad y datos; la capacidad viene del base.
