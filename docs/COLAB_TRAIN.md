# Colab train

## Path

Repo folder is often lowercase:

```text
/content/numi-ot.01
```

## Stack (avoid HybridCache error)

```python
!pip install -q --upgrade "transformers>=4.46.0,<4.52.0" "peft>=0.13.0,<0.16.0" "accelerate>=0.34.0" "bitsandbytes>=0.43.0"
# Runtime → Restart session
```

```python
%cd /content/numi-ot.01
!git pull origin main
!pip install -q -r requirements-lora.txt
!pip install -q -e .

import transformers, peft
print(transformers.__version__, peft.__version__)  # 4.51.x + 0.15.x OK
```

## Train (default — works on Colab)

```python
!python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

Base: **HuggingFaceTB/SmolLM2-1.7B-Instruct** (compatible with transformers 4.51).

## Hunyuan 7B

Needs transformers that register `hunyuan_v1_dense` (newer than 4.51). That often fights peft on Colab.

```python
# only if you know your stack supports Hunyuan:
!python scripts/training/train_base_lora.py --config configs/training/lora_frontier_hunyuan.yaml
```

## Warnings you can ignore

- `gradio` / `diffusers` vs `huggingface-hub` — unrelated to LoRA train
- `cudart_stub` / no GPU — train will be **very** slow on CPU; use a GPU runtime

## GPU

Runtime → Change runtime type → **T4 or better**.
