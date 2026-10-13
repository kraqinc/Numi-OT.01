# Base potente e independiente (Numi)

**Prohibido en producto:** Microsoft Phi, Qwen, Anthropic, OpenAI como identidad.

## Default (recomendado)

| Config | Base | Notas |
|--------|------|--------|
| `lora_frontier.yaml` | **`tencent/Hunyuan-7B-Instruct`** | 7B independiente, fuerte, open |
| `lora_large.yaml` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Más reasoning; linaje Llama |

Identidad de producto siempre **Numi OT** vía system prompt + SFT.

## Colab / GPU

```python
%cd /content/Numi-OT.01
!git pull
!pip install -q -r requirements-lora.txt && pip install -q -e .

!python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml

!python scripts/export_lora_safetensors.py --adapter outputs/lora_frontier/final

from serving.base_engine import BaseLoRAEngine
e = BaseLoRAEngine.from_adapter(
    "tencent/Hunyuan-7B-Instruct",
    "outputs/lora_frontier/final",
    load_in_4bit=True,
)
print(e.generate([{"role":"user","content":"¿Qué modelo eres?"}], max_tokens=64).text)
```

GPU: L4/A100 ideal. T4 puede ir justo en 4-bit.

## Render

```text
NUMI_BACKEND=peft
NUMI_PEFT_BASE=tencent/Hunyuan-7B-Instruct
NUMI_PEFT_ADAPTER=<tu-adapter-hf-o-ruta>
NUMI_API_KEY=<tu key Termux>
```

Plan con **mucha RAM** o instancia GPU. 7B en CPU puro es lento.
