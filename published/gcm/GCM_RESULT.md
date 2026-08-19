# Frozen Weather GCM result

The final TDN-specific train-only temporal GCM contract conditions each tested variable on target-history lags and all other observed covariates at the same historical slice. The supported lag is discovery evidence; TDN consumes the complete observed history of each selected physical variable.

| Rank | Feature | Supported lag | Conditional residual effect | BH q-value | Stability |
|---:|---|---:|---:|---:|---:|
| 1 | `Tlog_degC` | 2 | 0.186608 | 1.448e-04 | 1.00 |
| 2 | `H2OC_m` | 48 | -0.148329 | 1.128e-02 | 1.00 |
| 3 | `VPdef_mbar` | 95 | -0.129909 | 1.669e-03 | 1.00 |
| 4 | `rho_g` | 0 | 0.073366 | 4.686e-02 | 1.00 |
| 5 | `PAR_ol` | 95 | -0.069137 | 2.225e-02 | 1.00 |
| 6 | `Tdew_degC` | 0 | -0.210467 | 1.946e-02 | 0.75 |
| 7 | `p_mbar` | 0 | -0.092405 | 2.264e-02 | 0.75 |
| 8 | `max_PAR` | 95 | -0.071566 | 1.669e-03 | 0.75 |
| 9 | `raining_s` | 95 | -0.070886 | 1.128e-02 | 0.75 |
| 10 | `wv_m` | 95 | 0.066662 | 7.484e-03 | 0.75 |

## Frozen temporal-testing settings

- screened training origins: 36,000;
- forward residual blocks: 5;
- HAC/Newey–West bandwidth: 95 (matching the overlap range induced by the 96-step future-mean response);
- BH-FDR: 0.05;
- minimum conditional residual effect: 0.015;
- temporal stability: 0.60;
- maximum one lag per physical variable; maximum 11 variables (10 satisfy all gates under the final contract).

## Interpretation boundary

A retained pair is a **lagged causal-driver candidate under the observed-variable and temporal-testing assumptions**. It remains conditionally relevant after accounting for target-history lags and the other observed covariates at the tested historical slice. This is not a direct-parent proof, complete causal graph, or intervention-effect estimate.

## Data-quality handling

The supplied Weather training partition contains the benchmark missing-value sentinel `-9999` in a small number of rows (`wv_m`: 1, `max_PAR`: 30, `OT`: 50). For **discovery and selector construction only**, the sentinel is treated as missing and replaced by strictly backward-looking forward fill within the training partition. Validation/test rows are never used by discovery, and the forecasting benchmark CSV is not modified.

Manifest: `published/gcm/weather_gcm_manifest.json`
