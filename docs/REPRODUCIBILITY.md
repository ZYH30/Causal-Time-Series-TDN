# Reproducibility

The repository separates two reproducibility targets.

## Published-result reproduction

`bash scripts/reproduce_published_results.sh` never trains a model. It verifies the canonical artifact hashes, loads the frozen TDN checkpoints, reruns test-set inference, and regenerates the original-scale tables. The H96 selector study is also recomputed from its frozen checkpoints.

The 14 forecasting baselines are regenerated from the frozen v1.4.1 per-seed benchmark metrics because baseline checkpoints were not retained in the canonical release artifact. This distinction is explicit in the generated report.

## Train from scratch

`bash scripts/train_from_scratch.sh` creates a new directory under `results/from_scratch/` and never writes into `published/`. Final TDN and selector-study runs use `repro-mode=strict`, which sets Python/NumPy/PyTorch seeds, deterministic cuDNN behavior, deterministic PyTorch algorithms, a deterministic cuBLAS workspace, math-only scaled-dot-product attention, an explicit DataLoader generator, and TF32-disabled execution.

Baselines preserve the frozen v1.4.1 `run_seeded.py` implementation and paper hyperparameters. `baseline/run_strict_seeded.py` is an execution wrapper that enables deterministic PyTorch/CUDA behavior before delegating to the unchanged seeded runner; no baseline is retuned. Consequently, training-from-scratch values can vary slightly across GPU models, CUDA libraries, or PyTorch builds. Published numerical claims should be verified through the frozen-artifact path.

## Frozen environment

The experiments were finalized on:

- Python 3.12.7
- PyTorch 2.6.0+cu124
- CUDA 12.4
- NVIDIA A800 80GB PCIe
- 8 GPUs for the full experiment queue

The canonical Weather file SHA256 is:

`f365909fa07b621e61a9a1293dc69ae803d3a865905066b481e7dcc6819932d7`

The three final seeds are `20260810`, `20260811`, and `20260812`.
