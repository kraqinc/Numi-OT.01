# Colab — 32B full send

## 1) GPU

Runtime → Change runtime type → **GPU** (T4 or better).

```python
!nvidia-smi
```

## 2) Stack (HybridCache fix)

```python
!pip install -q --upgrade "transformers>=4.46.0,<4.52.0" "peft>=0.13.0,<0.16.0" "accelerate>=0.34.0" "bitsandbytes>=0.43.0"
```

**Runtime → Restart session**, then:

```python
%cd /content/numi-ot.01
!git pull origin main
!pip install -q -r requirements-lora.txt
!pip install -q -e .

import transformers, peft, torch
print(transformers.__version__, peft.__version__, torch.cuda.is_available())
```

## 3) Train 32B

```python
!python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

Expect:

```text
Base model: Qwen/Qwen2.5-32B-Instruct
...
Training complete → outputs/lora_frontier/final
```

## OOM (out of memory)?

```python
!python scripts/training/train_base_lora.py --config configs/training/lora_frontier_14b.yaml
```

## Save adapter

Download / push `outputs/lora_frontier/final` (adapter weights + tokenizer).
Set on Render:

```text
NUMI_BACKEND=peft
NUMI_PEFT_BASE=Qwen/Qwen2.5-32B-Instruct
NUMI_PEFT_ADAPTER=<path or HF repo of your adapter>
NUMI_PEFT_4BIT=1
```
