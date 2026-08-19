# Causal Time-Series Forecasting with Temporal Driver Networks (TDN)

This repository provides the official implementation and reproducibility package for the Temporal Driver Network (TDN) framework used in our time-series forecasting experiments.

The released code includes temporal causal-driver discovery, TDN forecasting, baseline comparisons, controlled driver-selection experiments, and the frozen model checkpoints corresponding to the reported Weather benchmark results.

The repository supports two reproducibility paths:

1. **Published checkpoint reproduction** — directly evaluate the released checkpoints and regenerate the reported results without retraining.
2. **Training reproduction** — train the final models from scratch using the same data split, driver-selection contract, hyperparameters, random seeds, and training configuration used in the reported experiments.

---

## Repository Structure

```text
Causal-Time-Series-TDN/
├── dataset/
│   └── weather.csv
│
├── gcm/
│   └── Temporal causal-driver discovery
│
├── tdn/
│   └── Temporal Driver Network
│
├── baseline/
│   └── Baseline forecasting models
│
├── published/
│   ├── gcm/
│   ├── tdn/
│   │   ├── H96/
│   │   ├── H192/
│   │   └── H720/
│   ├── selector_study/
│   ├── baseline/
│   └── original_scale/
│
├── scripts/
│   ├── reproduce_published_results.sh
│   ├── train_from_scratch.sh
│   ├── run_tdn_multiseed.py
│   ├── verify_release.py
│   └── preflight.py
│
├── results/
├── README.md
└── NOTICE.md
```

The `published/` directory contains the frozen artifacts associated with the reported experimental results. Training scripts write new outputs only to `results/from_scratch/` and do not overwrite the released checkpoints.

---

## Environment

The reported experiments were conducted using:

```text
Python 3.12.7
PyTorch 2.6.0
CUDA 12.4
8 × NVIDIA A800 80GB PCIe
```

A CUDA-enabled GPU is recommended for training and checkpoint evaluation.

Before running experiments, verify the repository and environment:

```bash
python scripts/verify_release.py
python scripts/preflight.py
```

---

## Dataset

The public experiments use the Weather dataset distributed with this repository:

```text
dataset/weather.csv
```

The dataset identity used for the reported experiments is:

```text
SHA256:
f365909fa07b621e61a9a1293dc69ae803d3a865905066b481e7dcc6819932d7
```

The target variable is `OT`.

The train/validation/test processing and target normalization are implemented consistently across the released training and evaluation pipelines.

---

# 1. Published Checkpoint Reproduction

This is the recommended path for reproducing the numerical results reported in the paper.

The released checkpoints are the frozen models corresponding to the reported experiments. This reproduction path:

```text
loads frozen checkpoints
        ↓
runs checkpoint-only inference
        ↓
performs no model training
        ↓
evaluates on the frozen test split
        ↓
restores errors to the original OT scale
        ↓
regenerates the reported result tables
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda \
bash scripts/reproduce_published_results.sh
```

No optimization or retraining is performed.

This avoids training-time numerical variation caused by differences in GPU hardware, CUDA kernels, PyTorch versions, or other low-level execution details.

## Released TDN Checkpoints

The final TDN results use three random seeds:

```text
20260810
20260811
20260812
```

and three forecasting horizons:

```text
H = 96
H = 192
H = 720
```

The final forecasting architecture is:

```text
ar_b48_m050
```

with:

```text
seq_len = 336
decoder_mode = ar
rollout_block_size = 48
rollout_final_mix = 0.50
rollout_ramp_epochs = 4
adv_weight = 0.03
```

## Expected TDN Results

All values below are reported on the original `OT` scale.

| Horizon | MSE mean ± SD | MAE mean ± SD | RMSE mean ± SD |
|---:|---:|---:|---:|
| 96 | 181.858444 ± 17.610244 | 10.073039 ± 0.360220 | 13.475007 ± 0.651102 |
| 192 | 194.717555 ± 11.969416 | 10.565113 ± 0.487936 | 13.949674 ± 0.431515 |
| 720 | 295.003316 ± 8.992330 | 12.711693 ± 0.318355 | 17.174334 ± 0.261452 |

Per-seed TDN results are:

| Horizon | Seed | MSE | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 96 | 20260810 | 179.867798 | 10.229647 | 13.411480 |
| 96 | 20260811 | 200.379425 | 10.328447 | 14.155544 |
| 96 | 20260812 | 165.328110 | 9.661022 | 12.857998 |
| 192 | 20260810 | 204.806503 | 10.964467 | 14.311062 |
| 192 | 20260811 | 181.492203 | 10.021243 | 13.471904 |
| 192 | 20260812 | 197.853958 | 10.709628 | 14.066057 |
| 720 | 20260810 | 293.917114 | 12.838150 | 17.144011 |
| 720 | 20260811 | 304.489410 | 12.947390 | 17.449625 |
| 720 | 20260812 | 286.603424 | 12.349540 | 16.929366 |

The reproduction script also regenerates the baseline comparison and driver-selection results using the released experimental artifacts.

---

# 2. Training Reproduction

The complete training pipeline is also provided for reproducing the experimental procedure from scratch.

The training configuration follows the execution contract used for the reported experiments, including:

```text
Weather data split
GCM-selected driver view
model architecture
optimizer configuration
training schedule
random seeds
forecast horizons
rollout-aware training
checkpoint-selection rule
```

The three default random seeds are:

```text
20260810
20260811
20260812
```

All newly trained models are written to:

```text
results/from_scratch/
```

and never overwrite the released checkpoints under `published/`.

## Full Training Pipeline

Run:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/train_from_scratch.sh
```

The pipeline includes:

```text
Temporal GCM driver discovery
        ↓
final driver manifests
        ↓
TDN H96 / H192 / H720 training
        ↓
three-seed evaluation
        ↓
baseline experiments
        ↓
driver-selection study
        ↓
original-scale result aggregation
```

The baseline models retain their original model architectures and training hyperparameters. Random-seed control is added without retuning the baseline configurations.

## Reproducing H96 TDN Only

To train only the final H96 TDN with the three reported random seeds:

```bash
RUN_ROOT=results/from_scratch/h96_$(date +%Y%m%d_%H%M%S)

mkdir -p "$RUN_ROOT"

nohup python -u scripts/run_tdn_multiseed.py \
  --selector published/gcm/selector_gcm.json \
  --horizons 96 \
  --seeds 20260810,20260811,20260812 \
  --gpus 0,1,2 \
  --repro-mode legacy \
  --output-dir "$RUN_ROOT/runs" \
  --log-dir "$RUN_ROOT/logs" \
  --tag h96_paper_reproduction \
  --report "$RUN_ROOT/TDN_H96_ThreeSeed_OriginalScale_Report.md" \
  > "$RUN_ROOT/launcher.log" 2>&1 &
```

The `legacy` execution mode corresponds to the numerical training path used for the reported experiments.

Monitor the launcher with:

```bash
tail -f "$RUN_ROOT/launcher.log"
```

Monitor GPU utilization with:

```bash
watch -n 2 nvidia-smi
```

The final report is written to:

```text
$RUN_ROOT/TDN_H96_ThreeSeed_OriginalScale_Report.md
```

A representative three-seed training reproduction on the original experimental environment produced:

```text
MSE  = 174.970678 ± 12.350316
MAE  = 10.026475 ± 0.410574
RMSE = 13.222090 ± 0.469585
```

which is consistent with the performance level of the released paper checkpoints.

Training from scratch may exhibit small numerical differences across GPU models, CUDA versions, PyTorch versions, and low-level kernel implementations. For exact reproduction of the numerical values reported in the paper, use the **Published Checkpoint Reproduction** path.

---

## Temporal Driver Discovery

The GCM component identifies variable-lag pairs that provide the causal-driver view used by TDN.

The released GCM-10 driver set is:

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

These variable-lag pairs should be interpreted as temporal causal-driver candidates under the observed-variable and temporal-testing assumptions of the framework. They are not intended to represent a complete causal graph, direct causal parents, or interventional effect estimates.

---

## Driver-Selection Study

The controlled H96 driver-selection experiment compares five 10-variable input views:

```text
GCM-10
Correlation-10
Mutual-Information-10
Random-10 #0
Random-10 #1
```

All conditions use the same:

```text
forecasting architecture
optimization configuration
random seeds
forecast horizon
number of selected physical variables
```

The experiment isolates the effect of the driver-selection strategy while keeping the forecasting model unchanged.

---

## Baselines

The Weather benchmark includes the following forecasting baselines:

```text
Autoformer
Crossformer
iTransformer
MICN
MultiPatchFormer
Nonstationary Transformer
PatchTST
Pyraformer
SegRNN
TimeMixer
TimesNet
TimeXer
Transformer
TSMixer
```

Baseline architectures and training hyperparameters follow their original benchmark configurations.

All reported baseline results use the same three random seeds:

```text
20260810
20260811
20260812
```

and horizons:

```text
96
192
720
```

---

## Result Scale

The Weather experiments internally optimize standardized targets.

All paper-facing metrics are converted back to the original `OT` scale.

The target scaler is fitted on the training partition only.

For the released Weather split:

```text
training target mean  = 411.054341638
training target scale = 383.957097966
```

The conversion is:

```text
MAE_original  = MAE_standardized  × scale
RMSE_original = RMSE_standardized × scale
MSE_original  = MSE_standardized  × scale²
```

All main tables in this repository report metrics on the original scale.

---

## Artifact Integrity

The release contains SHA256 records for the frozen dataset and published checkpoints.

Run:

```bash
python scripts/verify_release.py
```

to verify the released artifacts before evaluation.

The Weather dataset SHA256 is:

```text
f365909fa07b621e61a9a1293dc69ae803d3a865905066b481e7dcc6819932d7
```

The released checkpoints under `published/` should be treated as immutable paper artifacts.

New training runs should always be stored under:

```text
results/from_scratch/
```

---

## Recommended Reproduction Workflow

For readers who want to verify the reported results:

```bash
python scripts/verify_release.py
python scripts/preflight.py

CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda \
bash scripts/reproduce_published_results.sh
```

For readers who want to retrain the models:

```bash
python scripts/verify_release.py
python scripts/preflight.py

GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/train_from_scratch.sh
```

---

## Citation

If this repository is useful for your research, please cite the corresponding paper.

```bibtex
@inproceedings{TDN2026,
  title     = {Temporal Driver Networks for Causal-Aware Time-Series Forecasting},
  author    = {Zhao, Yonghe et al.},
  year      = {2026}
}
```

---

## License

Please refer to the repository license and the licenses of the included baseline implementations and datasets before redistribution or commercial use.
