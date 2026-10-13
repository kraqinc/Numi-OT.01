#!/usr/bin/env bash
# Copy Colab/local PEFT adapter into the path Docker expects.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-outputs/lora_frontier/final}"
DST="adapters/final"

if [[ ! -d "$SRC" ]]; then
  echo "Missing adapter dir: $SRC"
  echo "Train first or unzip release into that path."
  exit 1
fi

rm -rf "$DST"
mkdir -p "$DST"
cp -a "$SRC"/. "$DST"/
echo "Bundled $SRC → $DST"
ls -la "$DST"
