# Base models (quality vs hardware)

Tiny 1–4B models feel weak in real chat. Use size that matches GPU budget.

| Config | Base | VRAM (4-bit LoRA) | Where |
|--------|------|-------------------|--------|
| `lora_frontier_14b.yaml` | Qwen2.5-**14B**-Instruct | ~12–16 GB | Colab T4 / 16GB |
| **`lora_frontier.yaml`** | Qwen2.5-**32B**-Instruct | ~20–24 GB | 4090 / A10 / L40 / A100 |
| `lora_frontier_72b.yaml` | Qwen2.5-**72B**-Instruct | ~40–80 GB | A100 80GB / multi-GPU |

After SFT the **product identity is Numi OT** (system prompt + datasets). The HF base is only the engine.

## Money path (recommended)

1. Rent **1× A100 40/80GB** or **RTX 4090 24GB** (RunPod, Lambda, Vast).
2. Install stack: `pip install -r requirements-lora.txt`
3. Train 32B:

```bash
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

4. Serve adapter on same class GPU; Flutter still talks to your API as Numi.

## Not free Colab

Free T4 often OOM on 32B. Use `lora_frontier_14b.yaml` there, or pay for A100.

## Hunyuan-7B

Needs `transformers>=4.56` / 5.x (`hunyuan_v1_dense`). Only 7B — weaker than 32B open instruct for general chat.
