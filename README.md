# TDN: Causal Driver Discovery and De-confounded Forecasting for Dynamic Temporal Data Pipelines

Official research code for **TDN**, a temporal forecasting framework that combines conditional-independence-based causal-driver discovery with a decoupled forecasting network. The repository contains the public Weather benchmark used in the manuscript, frozen checkpoints for the reported TDN results, the 14 forecasting baselines, the cardinality-matched driver-selection study, and the H96 systematic tuning pipeline.

The release supports two distinct workflows:

1. **Reproduce Published Results** — load frozen checkpoints, run test inference only, and regenerate the original-scale paper tables without retraining.
2. **Train from Scratch** — rebuild the GCM driver view and train the final TDN, baselines, and selector study in a new output directory under a strict deterministic execution contract.

## Method overview

TDN uses a two-stage forecasting pipeline. First, temporal GCM screens the observed Weather covariates on the chronological training partition and returns a compact set of statistically supported variable-lag driver candidates. These selected variables define the historical driver view used by the forecasting model; the selected screening lag is evidence used during discovery and is not interpreted as a mandatory shifted input at forecast time.

The forecasting stage maintains separate driver-history and target-history pathways. An adversarial reconstruction objective reduces recoverable target-history summary information in the driver representation before late fusion. The final public model uses the `ar_b48_m050` rollout-aware configuration: pure autoregressive decoding, 48-step rollout blocks during training, final rollout mix `0.50`, and a four-epoch rollout ramp.

The causal claim is deliberately bounded: GCM identifies candidate causal drivers under the observed-variable and temporal-testing assumptions encoded by the discovery procedure. The released code does not treat the selected variables as a complete causal graph, direct causal parents, or intervention-effect estimates.

## Repository structure

```text
.
├── gcm/                         # Temporal GCM discovery and selector construction
├── tdn/                         # Final TDN model and training implementation
├── baseline/                    # 14 paper-facing forecasting baselines
├── dataset/
│   └── weather.csv              # Public Weather benchmark
├── published/                   # Immutable canonical research artifacts
│   ├── tdn/                     # H96/H192/H720 frozen TDN checkpoints
│   ├── selector_study/          # Selector manifests + frozen non-GCM H96 checkpoints
│   ├── baseline/                # Frozen baseline benchmark report
│   ├── original_scale/          # Paper-facing original-scale tables and audits
│   ├── gcm/                     # Frozen GCM manifest and GCM-10 selector
│   ├── tuning/                  # Frozen H96 tuning report
│   ├── CANONICAL_PROVENANCE.json
│   └── SHA256SUMS
├── scripts/
│   ├── reproduce_published_results.sh
│   ├── train_from_scratch.sh
│   ├── run_gcm_fresh.sh
│   ├── run_hyperparameter_tuning.sh
│   └── verify_release.py
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── PROVENANCE.md
└── results/                     # Fresh outputs only
```

## Environment

The frozen experiment environment was:

```text
Python       3.12.7
PyTorch      2.6.0+cu124
CUDA         12.4
GPU          NVIDIA A800 80GB PCIe
GPU count    8 for the full experiment queue
```

Install a CUDA-enabled PyTorch build appropriate for the target system, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Before running experiments, verify the release:

```bash
python scripts/verify_release.py
```

For a lightweight code/data check that does not require training:

```bash
python scripts/preflight.py
```

Optional release tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

To validate the complete train-from-scratch launch graph without starting any experiment:

```bash
DRY_RUN=1 GPU_IDS=0,1,2 bash scripts/train_from_scratch.sh
```

## Dataset

The repository contains the exact public Weather CSV used by all reported public experiments:

```text
dataset/weather.csv
rows: 52696
columns: 22
target: OT
SHA256: f365909fa07b621e61a9a1293dc69ae803d3a865905066b481e7dcc6819932d7
```

The chronological split is 70% train, 10% validation, and 20% test. Scalers are fit on the first 70% only. The target scaler used for paper-facing error restoration has:

```text
training rows: 36887
OT mean:       411.054341638
OT scale:      383.957097966
```

All tables below and all reproduction reports use the original `OT` scale.

The industrial dataset described in the manuscript is not redistributed in this repository because it is subject to confidentiality and data-governance restrictions. The public release therefore provides a complete reproduction path for the Weather benchmark.

## Reproduce published results

This is the recommended path for verifying the numerical results in the manuscript. It performs **no optimization and no checkpoint update**.

```bash
CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda \
bash scripts/reproduce_published_results.sh
```

The workflow performs four operations:

1. verifies the published artifact and checkpoint hashes;
2. loads the nine final TDN checkpoints and reruns test-set inference for H96/H192/H720;
3. loads the 15 H96 selector-study checkpoints and reruns test-set inference;
4. regenerates the original-scale benchmark and selector tables.

Outputs are written to:

```text
results/reproduce_published/
├── main_checkpoint_inference.json
├── selector_checkpoint_inference.json
├── Weather_Main_Benchmark_Reproduced.md
├── Weather_Main_Benchmark_Reproduced.csv
├── Weather_Selector_Study_Reproduced.md
└── Weather_Selector_Study_Reproduced.csv
```

The 14 baseline rows in the regenerated main benchmark are derived from the frozen three-seed v1.4.1 benchmark metrics because baseline checkpoints were not retained in the canonical release artifact. No baseline is retrained during published-result reproduction. TDN and selector-study rows are recomputed directly from frozen checkpoints. The GCM-10 selector condition reuses the same three frozen H96 TDN checkpoints used by the main benchmark.

### Expected final TDN results

| Horizon | MSE mean ± SD | MAE mean ± SD | RMSE mean ± SD |
|---:|---:|---:|---:|
| 96 | 181.858444 ± 17.610244 | 10.073039 ± 0.360220 | 13.475007 ± 0.651102 |
| 192 | 194.717555 ± 11.969416 | 10.565113 ± 0.487936 | 13.949674 ± 0.431515 |
| 720 | 295.003316 ± 8.992330 | 12.711693 ± 0.318355 | 17.174334 ± 0.261452 |

The full 15-model benchmark is available in `published/original_scale/Weather_OriginalScale_Final_Report.md`.

### H96 cardinality-matched driver selection

| Selector | MSE mean ± SD | MAE mean ± SD | RMSE mean ± SD |
|---|---:|---:|---:|
| GCM-10 | 181.858444 ± 17.610244 | 10.073039 ± 0.360220 | 13.475007 ± 0.651102 |
| Correlation-10 | 1149.502655 ± 1705.652312 | 10.022637 ± 0.785973 | 27.172404 ± 24.834344 |
| MI-10 | 293108.436620 ± 507379.386833 | 48.491476 ± 66.775577 | 321.268558 ± 533.706310 |
| Random-10 #0 | 68357.693975 ± 109812.089297 | 13.958046 ± 4.948753 | 184.854749 ± 226.450046 |
| Random-10 #1 | 70316.960749 ± 121490.042455 | 27.249842 ± 29.839803 | 161.781139 ± 257.324184 |

The paper-facing comparison uses the three-seed aggregate mean and sample standard deviation. GCM-10 yields the best aggregate MSE and a substantially more stable cross-seed error profile; MAE is close to the correlation selector.

## Train from scratch

The complete public training workflow creates a **new run directory** and never writes into `published/`:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/train_from_scratch.sh
```

A typical output root is:

```text
results/from_scratch/20260819T120000Z/
├── gcm/
├── selectors/
├── tdn/
├── baselines/
├── selector_study/
└── Weather_Final_OriginalScale_Benchmark_Report.md
```

The final TDN and selector-study launches use the strict TDN reproducibility contract: deterministic PyTorch algorithms, deterministic cuBLAS/cuDNN behavior, an explicit train-DataLoader generator, math-only scaled-dot-product attention, fixed Python/NumPy/PyTorch/CUDA seeds, and disabled TF32 execution. Baselines use the unchanged frozen `run_seeded.py` through `run_strict_seeded.py`, which adds deterministic PyTorch/CUDA execution without changing any paper-facing architecture or optimization setting.

The three seeds are:

```text
20260810
20260811
20260812
```

The final TDN contract is:

```text
seq_len=336
batch_size=128
driver_hidden=64
target_hidden=64
target_embedding=16
calendar_hidden=32
num_layers=2
driver_heads=4
target_heads=8
dropout=0.05
adv_weight=0.03
decoder_mode=ar
rollout_block_size=48
rollout_final_mix=0.50
rollout_ramp_epochs=4
```

Training from scratch is intended to reproduce the experimental procedure. Bitwise identity to historical checkpoints is not assumed across different GPU models, CUDA kernels, or PyTorch builds; the canonical numerical verification path is the frozen-checkpoint workflow above.

## Individual experiment entry points

### Temporal GCM discovery

Create a fresh run root and build the GCM/selector manifests:

```bash
bash scripts/run_gcm_fresh.sh results/from_scratch/gcm_example
```

The frozen GCM-10 view contains the following ten physical variables with their supported screening lags:

```text
Tlog_degC   lag 2
H2OC_m      lag 48
VPdef_mbar  lag 95
rho_g       lag 0
PAR_ol      lag 95
Tdew_degC   lag 0
p_mbar      lag 0
max_PAR     lag 95
raining_s   lag 95
wv_m        lag 95
```

### Final TDN only

After building a selector manifest, the final three-seed launcher can be called directly:

```bash
python scripts/run_tdn_multiseed.py \
  --selector results/from_scratch/gcm_example/selectors/selector_gcm.json \
  --horizons 96,192,720 \
  --seeds 20260810,20260811,20260812 \
  --gpus 0,1,2 \
  --repro-mode strict \
  --output-dir results/from_scratch/tdn_only/runs \
  --log-dir results/from_scratch/tdn_only/logs \
  --tag tdn_only \
  --report results/from_scratch/tdn_only/TDN_Report.md
```

### Baselines only

```bash
python scripts/run_baselines_multiseed.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --repro-mode strict \
  --output-root results/from_scratch/baselines_only/runs \
  --log-root results/from_scratch/baselines_only/logs \
  --run-tag baselines_only
```

The included paper-facing baselines are Autoformer, Crossformer, iTransformer, MICN, MultiPatchFormer, Nonstationary Transformer, PatchTST, Pyraformer, SegRNN, TimeMixer, TimesNet, TimeXer, Transformer, and TSMixer. Their architecture and training settings are retained from the frozen benchmark. The strict wrapper adds deterministic PyTorch/CUDA execution around the unchanged seeded runner; it does not retune a baseline.

### H96 systematic hyperparameter search

The systematic H96 search is intentionally retained as the original **test-aware** experimental protocol. It evaluates validation and test for every candidate and uses test MSE during stage promotion. It should therefore be interpreted as a tuning study, not as an untouched holdout estimate.

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/run_hyperparameter_tuning.sh
```

The frozen selected backbone was `s1_a36__p10_short_warm`.

## Published artifact integrity

`published/CANONICAL_PROVENANCE.json` records the checkpoint SHA256 values, expected metrics, and source hashes for the canonical public artifacts. `published/SHA256SUMS` verifies the exact public artifact files shipped in this repository. The original canonical handoff archive hash is recorded in `docs/PROVENANCE.md`.

Run:

```bash
python scripts/verify_release.py
```

before reproducing results or publishing a modified fork.

## License

The baseline library is derived from Time-Series-Library and retains its MIT license. See `LICENSE` and `NOTICE.md`.

## Citation

If this repository is useful for your research, please cite the accompanying manuscript:

```bibtex
@misc{zhao2026tdn,
  title   = {TDN: Causal Driver Discovery and De-confounded Forecasting for Dynamic Temporal Data Pipelines},
  author  = {Zhao, Yonghe and Ma, Chao and Li, Jiawei and Cui, Yangyang and Wang, Yuezhu and Sun, Huiyan},
  year    = {2026},
  note    = {Manuscript}
}
```
