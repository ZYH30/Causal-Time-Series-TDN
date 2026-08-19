#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-results/reproduce_published}
mkdir -p "$OUT"
python scripts/verify_release.py
python scripts/reproduce/evaluate_checkpoints.py --scope main --device "$DEVICE" --output "$OUT/main_checkpoint_inference.json"
python scripts/reproduce/evaluate_checkpoints.py --scope selector --device "$DEVICE" --output "$OUT/selector_checkpoint_inference.json"
python scripts/reproduce/build_published_tables.py \
  --main-inference "$OUT/main_checkpoint_inference.json" \
  --selector-inference "$OUT/selector_checkpoint_inference.json" \
  --output-dir "$OUT"
echo "[DONE] Published results reproduced without retraining: $OUT"
