# Numi real: LoRA sobre base open

## Default

`google/gemma-2-9b-it` — funciona con 4-bit + LoRA en Colab GPU.

**No uses** `openai/gpt-oss-20b` con `load_in_4bit: true`: viene pre-quantizado (Mxfp4) y choca con BitsAndBytes.

## Colab

```python
%cd Numi-OT.01
!git pull
!pip install -q -r requirements-lora.txt
!pip install -q -e .

# Acepta licencia Gemma en HF si pide login:
# from huggingface_hub import login; login()

!python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

Si Gemma es gated:

```python
from huggingface_hub import login
login()  # token con acceso a google/gemma-2-9b-it
```

## Inferencia (después del train)

```python
from serving.base_engine import BaseLoRAEngine

e = BaseLoRAEngine.from_adapter(
    base_model_id="google/gemma-2-9b-it",
    adapter_path="outputs/lora_frontier/final",
    load_in_4bit=True,
)
print(e.generate([{"role": "user", "content": "¿Qué modelo eres?"}]).text)
```

Si el adapter no existe → error claro `FileNotFoundError` (entrena primero).

## Otros bases

```bash
# DeepSeek (si el id existe en el hub y te cabe en VRAM)
python scripts/training/train_base_lora.py --model-id deepseek-ai/DeepSeek-V2-Lite-Chat
```
