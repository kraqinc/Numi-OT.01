# Numi OT LoRA v0.2 (SmolLM2-1.7B base)

## Identity check (Colab generate)

> Numi OT. No modelo de vendor o engine. Numi AI / Numi OT.

## Files (upload as GitHub Release assets — not in git)

| File | Description |
|------|-------------|
| `numi_lora.safetensors` | LoRA adapter weights (336 tensors) |
| `final/` | Full PEFT dir (`adapter_config.json`, tokenizer, …) |
| `numi_lora.meta.json` | Export metadata |

**Base model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`  
**Trainable:** ~18.1M params (1.05%)

## Load

```python
from serving.base_engine import BaseLoRAEngine

e = BaseLoRAEngine.from_adapter(
    base_model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    adapter_path="path/to/final",  # or download release + unzip
    load_in_4bit=True,
)
print(e.generate([{"role": "user", "content": "¿Qué modelo eres?"}]).text)
```

## Publish from Colab

```python
# 1) Zip adapter
!cd /content/Numi-OT.01/outputs/lora_frontier && zip -r numi-lora-v0.2.zip final numi_lora.safetensors numi_lora.meta.json

# 2) Create GitHub Release + upload (needs GH token with repo scope)
import os
os.environ["GH_TOKEN"] = "ghp_..."  # PAT
!gh release create v0.2-lora \
  /content/Numi-OT.01/outputs/lora_frontier/numi-lora-v0.2.zip \
  --repo kraqinc/Numi-OT.01 \
  --title "Numi OT LoRA v0.2" \
  --notes "SmolLM2-1.7B + Numi identity. Identity: Numi OT."
```

Or Hugging Face Hub:

```python
from huggingface_hub import HfApi, login
login()
api = HfApi()
api.create_repo("kraqinc/numi-ot-lora-v0.2", private=True, exist_ok=True)
api.upload_folder(
    folder_path="/content/Numi-OT.01/outputs/lora_frontier/final",
    repo_id="kraqinc/numi-ot-lora-v0.2",
    repo_type="model",
)
api.upload_file(
    path_or_fileobj="/content/Numi-OT.01/outputs/lora_frontier/numi_lora.safetensors",
    path_in_repo="numi_lora.safetensors",
    repo_id="kraqinc/numi-ot-lora-v0.2",
)
```
