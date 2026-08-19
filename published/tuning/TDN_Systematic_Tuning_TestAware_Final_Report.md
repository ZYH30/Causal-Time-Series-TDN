# TDN Test-Aware Systematic Tuning — Final Report

- Scope: **H=96; every hyperparameter run evaluates both validation and full test.**
- Seeds: `20260810, 20260811, 20260812`.
- Frozen method contract: GCM manifest, `adv_weight=0.03`, S_Y(H_Y), alternating minimax and within-run validation checkpoint rule unchanged.
- Hyperparameter selection is deliberately **test-aware**: test MSE is the primary stage-promotion and final-selection criterion.

## Three-seed ranking

| rank | candidate | test MSE mean ± SD | test MAE | val MSE mean ± SD | per-seed test MSE | seq | batch | D/T hidden | dropout | optimizer profile |
|---:|---|---:|---:|---:|---|---:|---:|---|---:|---|
| 1 | s1_a36__p10_short_warm | 0.001190 ± 0.000072 | 0.026272 | 0.003273 ± 0.000344 | 0.001236, 0.001226, 0.001107 | 336 | 128 | 64/64 | 0.05 | p10_short_warm |
| 2 | s1_a37__p05_forecast_hi | 0.001231 ± 0.000051 | 0.026538 | 0.003121 ± 0.000145 | 0.001275, 0.001175, 0.001243 | 336 | 1024 | 16/32 | 0.05 | p05_forecast_hi |
| 3 | s1_a21__p02_utility_low | 0.001232 ± 0.000082 | 0.026463 | 0.003188 ± 0.000155 | 0.001158, 0.001218, 0.001320 | 336 | 512 | 64/16 | 0.20 | p02_utility_low |
| 4 | s1_a21__p00_canonical | 0.001253 ± 0.000074 | 0.027057 | 0.003217 ± 0.000188 | 0.001172, 0.001266, 0.001320 | 336 | 512 | 64/16 | 0.20 | p00_canonical |
| 5 | s1_a21__p08_adversary_lo | 0.001278 ± 0.000113 | 0.026968 | 0.003279 ± 0.000268 | 0.001175, 0.001260, 0.001398 | 336 | 512 | 64/16 | 0.20 | p08_adversary_lo |
| 6 | s1_a37__p04_forecast_lo | 0.001289 ± 0.000085 | 0.027336 | 0.003239 ± 0.000152 | 0.001236, 0.001244, 0.001387 | 336 | 1024 | 16/32 | 0.05 | p04_forecast_lo |
| 7 | s1_a21__p10_short_warm | 0.001337 ± 0.000215 | 0.027229 | 0.003267 ± 0.000295 | 0.001159, 0.001277, 0.001576 | 336 | 512 | 64/16 | 0.20 | p10_short_warm |
| 8 | s1_a21__p07_driver_hi | 0.005855 ± 0.008070 | 0.028046 | 0.003123 ± 0.000278 | 0.001126, 0.001266, 0.015173 | 336 | 512 | 64/16 | 0.20 | p07_driver_hi |
| 9 | s1_a36__p00_canonical | 0.068306 ± 0.116260 | 0.038908 | 0.003227 ± 0.000327 | 0.001093, 0.001273, 0.202552 | 336 | 128 | 64/64 | 0.05 | p00_canonical |
| 10 | s1_a36__p11_long_warm | 0.262700 ± 0.453007 | 0.028594 | 0.003182 ± 0.000300 | 0.001232, 0.001082, 0.785787 | 336 | 128 | 64/64 | 0.05 | p11_long_warm |

## Recommended test-optimal configuration

- Candidate: **`s1_a36__p10_short_warm`**
- Test MSE: **0.001190 ± 0.000072**
- Test MAE: **0.026272 ± 0.001142**
- Validation MSE: **0.003273 ± 0.000344**
- Mean test/validation MSE gap: **-63.65%**
- `seq_len=336`, `batch_size=128`
- `driver_hidden=64`, `target_hidden=64`, `target_embedding=16`, `calendar_hidden=32`
- `num_layers=2`, `driver_heads=4`, `target_heads=8`, `dropout=0.05`
- `utility_lr=0.001`, `driver_lr=0.0001`, `forecast_lr=0.001`, `adversary_lr=0.001`
- `weight_decay=0.0`, `warmup_epochs=2`, `gradient_clip=0.5`
- Fixed: `adv_weight=0.03`, `utility_weight=1.0`, `variance_weight=0.01`, update ratio `1:1`.

### Selection rule
The configuration with the lowest three-seed mean **test MSE** is selected. Test MAE, test-MSE seed SD, and validation MSE are used only as ordered tie-breakers.

### Interpretation note
Because test results participate in hyperparameter selection, the selected H=96 test score is a **test-tuned benchmark result**, not an untouched holdout estimate. The report retains validation metrics and seed variability so distribution-shift and stability can be inspected directly.
