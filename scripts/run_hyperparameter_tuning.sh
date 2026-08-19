#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
TUNE_ROOT=${TUNE_ROOT:-results/from_scratch/tuning_${RUN_ID}}
case "$TUNE_ROOT" in *published*) echo "Refusing to write inside published/." >&2; exit 2;; esac
mkdir -p "$TUNE_ROOT"
if [[ ! -s "$TUNE_ROOT/selectors/selector_gcm.json" ]]; then bash scripts/run_gcm_fresh.sh "$TUNE_ROOT"; fi
export TDN_TUNING_ROOT="$TUNE_ROOT/tuning"
python scripts/tuning/tuning_manager.py generate-stage1
python scripts/tuning/run_jobs.py --jobs "$TDN_TUNING_ROOT/stage1_jobs.json" --stage stage1 --gpus "$GPU_IDS" --selector "$TUNE_ROOT/selectors/selector_gcm.json"
python scripts/tuning/tuning_manager.py analyze-stage1
python scripts/tuning/tuning_manager.py generate-stage2
python scripts/tuning/run_jobs.py --jobs "$TDN_TUNING_ROOT/stage2_jobs.json" --stage stage2 --gpus "$GPU_IDS" --selector "$TUNE_ROOT/selectors/selector_gcm.json"
python scripts/tuning/tuning_manager.py analyze-stage2
python scripts/tuning/tuning_manager.py generate-stage3
python scripts/tuning/run_jobs.py --jobs "$TDN_TUNING_ROOT/stage3_jobs.json" --stage stage3 --gpus "$GPU_IDS" --selector "$TUNE_ROOT/selectors/selector_gcm.json"
python scripts/tuning/tuning_manager.py analyze-stage3
echo "[DONE] $TDN_TUNING_ROOT/TDN_Systematic_Tuning_TestAware_Final_Report.md"
