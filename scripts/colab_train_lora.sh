#!/usr/bin/env bash
# One-shot on Colab from /content/Numi-OT.01
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

pip install -q -r requirements-lora.txt
pip install -q -e .

MODEL_ID="${1:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"

python scripts/training/train_base_lora.py \
  --config configs/training/lora_frontier.yaml \
  --model-id "$MODEL_ID"

python scripts/export_lora_safetensors.py \
  --adapter outputs/lora_frontier/final \
  --output outputs/lora_frontier/numi_lora.safetensors

echo "OK: outputs/lora_frontier/final + numi_lora.safetensors"
