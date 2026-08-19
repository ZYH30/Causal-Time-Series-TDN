#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT_DIR=${1:?Usage: bash scripts/run_gcm_fresh.sh <fresh-run-root>}
case "$OUT_DIR" in *published*) echo "Refusing to write inside published/." >&2; exit 2;; esac
mkdir -p "$OUT_DIR/gcm" "$OUT_DIR/selectors"
python -m gcm.run_gcm --data dataset/weather.csv --output "$OUT_DIR/gcm/weather_gcm_manifest.json"
python -m gcm.selectors --data dataset/weather.csv --gcm-manifest "$OUT_DIR/gcm/weather_gcm_manifest.json" --output-dir "$OUT_DIR/selectors" --random-repeats 2 --seed 2026
echo "[DONE] $OUT_DIR/gcm and $OUT_DIR/selectors"
