#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-results/from_scratch/$RUN_ID}
case "$RUN_ROOT" in *published*) echo "Refusing to write training outputs inside published/." >&2; exit 2;; esac
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY RUN] No training will be started and no experiment output will be written."
  python scripts/run_tdn_multiseed.py \
    --selector published/gcm/selector_gcm.json \
    --horizons 96,192,720 --seeds 20260810,20260811,20260812 \
    --gpus "$GPU_IDS" --repro-mode strict \
    --output-dir "$RUN_ROOT/tdn/main" --log-dir "$RUN_ROOT/tdn/logs" \
    --tag "from_scratch_${RUN_ID}" --report "$RUN_ROOT/tdn/report.md" --dry-run
  python scripts/run_baselines_multiseed.py \
    --gpus "$GPU_IDS" --seeds 20260810,20260811,20260812 --horizons 96,192,720 \
    --output-root "$RUN_ROOT/baselines/runs" --log-root "$RUN_ROOT/baselines/logs" \
    --run-tag "from_scratch_${RUN_ID}" --repro-mode strict --dry-run
  for selector in gcm correlation mutual_information random_00 random_01; do
    python scripts/run_tdn_multiseed.py \
      --selector "published/selector_study/manifests/selector_${selector}.json" \
      --horizons 96 --seeds 20260810,20260811,20260812 \
      --gpus "$GPU_IDS" --repro-mode strict \
      --output-dir "$RUN_ROOT/selector_study/${selector}/runs" \
      --log-dir "$RUN_ROOT/selector_study/${selector}/logs" \
      --tag "selector_${RUN_ID}" --report "$RUN_ROOT/selector_study/${selector}/report.md" --dry-run
  done
  echo "[PASS] Train-from-scratch launch graph validated."
  exit 0
fi

if [[ -e "$RUN_ROOT" && "${ALLOW_EXISTING_RUN_ROOT:-0}" != "1" ]]; then
  echo "RUN_ROOT already exists: $RUN_ROOT" >&2
  echo "Use a new RUN_ID or set ALLOW_EXISTING_RUN_ROOT=1 explicitly." >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > results/from_scratch/LATEST_RUN.txt

# 1. Frozen temporal GCM + cardinality-matched selector manifests.
bash scripts/run_gcm_fresh.sh "$RUN_ROOT"

# 2. Final TDN: strict deterministic contract, fresh output only.
python scripts/run_tdn_multiseed.py \
  --selector "$RUN_ROOT/selectors/selector_gcm.json" \
  --horizons 96,192,720 --seeds 20260810,20260811,20260812 \
  --gpus "$GPU_IDS" --repro-mode strict \
  --output-dir "$RUN_ROOT/tdn/main" --log-dir "$RUN_ROOT/tdn/logs" \
  --tag "from_scratch_${RUN_ID}" \
  --report "$RUN_ROOT/tdn/TDN_Final_MultiSeed_OriginalScale_Report.md"

# 3. Frozen v1.4.1 baselines. Their paper protocol is retained; seeds and output IDs are isolated.
python scripts/run_baselines_multiseed.py \
  --gpus "$GPU_IDS" --seeds 20260810,20260811,20260812 --horizons 96,192,720 \
  --output-root "$RUN_ROOT/baselines/runs" --log-root "$RUN_ROOT/baselines/logs" \
  --run-tag "from_scratch_${RUN_ID}" --repro-mode strict

# 4. H96 cardinality-matched selector study, using the same strict TDN contract.
for selector in gcm correlation mutual_information random_00 random_01; do
  python scripts/run_tdn_multiseed.py \
    --selector "$RUN_ROOT/selectors/selector_${selector}.json" \
    --horizons 96 --seeds 20260810,20260811,20260812 \
    --gpus "$GPU_IDS" --repro-mode strict \
    --output-dir "$RUN_ROOT/selector_study/${selector}/runs" \
    --log-dir "$RUN_ROOT/selector_study/${selector}/logs" \
    --tag "selector_${RUN_ID}" \
    --report "$RUN_ROOT/selector_study/${selector}/report.md"
done

python scripts/build_main_report.py \
  --tdn-json "$RUN_ROOT/tdn/TDN_Final_MultiSeed_OriginalScale_Report.json" \
  --baseline-root "$RUN_ROOT/baselines/runs" \
  --output "$RUN_ROOT/Weather_Final_OriginalScale_Benchmark_Report.md"
python scripts/build_selector_report.py \
  --root "$RUN_ROOT/selector_study" \
  --output "$RUN_ROOT/selector_study/TDN_Selector_Study_OriginalScale_Report.md"

echo "[DONE] Train-from-scratch run: $RUN_ROOT"
echo "Published checkpoints under published/ were never modified."
