#!/usr/bin/env bash
# Train a smoke checkpoint and print the next integration steps.
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "==> Installing deps"
pip install -q -r requirements-dev.txt
pip install -q -e .

echo "==> Smoke SFT (CPU, tiny model)"
python scripts/training/train_sft.py \
  --config configs/training/sft_smoke.yaml \
  --device cpu

CKPT="outputs/sft_smoke/final.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found at $CKPT"
  exit 1
fi

echo ""
echo "OK: $CKPT"
echo ""
echo "Next:"
echo "  1. Local API test:"
echo "     export NUMI_ENV=development"
echo "     export NUMI_CONFIG=configs/serving/smoke.yaml"
echo "     export NUMI_CHECKPOINT=$CKPT"
echo "     python scripts/serve.py --host 127.0.0.1 --port 8000"
echo ""
echo "  2. curl:"
echo "     curl -s http://127.0.0.1:8000/live"
echo "     curl -s http://127.0.0.1:8000/v1/chat/completions \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hola\"}]}'"
echo ""
echo "  3. Deploy API (Render) with NUMI_API_KEY + same config/checkpoint"
echo "  4. Supabase: run schema.sql, deploy numi-chat, set secrets"
echo "  5. Flutter: use integrations/flutter/numi_client.dart"
echo ""
echo "Full guide: docs/INTEGRATION.md"
