#!/usr/bin/env python3
"""Build the paper-facing Weather causal-driver manifest for TDN.

Contract
--------
* discovery sees only the chronological training partition (first 70%);
* the response is the mean future OT over a fixed 96-step screening horizon;
* candidate lags and target-history conditioning lags are bounded by seq_len=96;
* each candidate is additionally conditioned on all other observed covariates at the same historical slice;
* forward residualization prevents later training blocks explaining earlier blocks;
* HAC/Newey-West uncertainty accounts for serial dependence in residual products;
* BH-FDR + residual-effect + temporal-stability gates determine selection;
* at most one supported lag is retained per physical variable;
* downstream TDN uses each selected variable's full observed history.  The selected
  lag is evidence for screening, not a future-value shift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcm.contract import discover_conditionally_relevant_lagged_drivers, save_manifest


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def segments_from_time(stamps: pd.Series, gap_multiplier: float = 3.0) -> np.ndarray:
    stamps = pd.to_datetime(stamps)
    differences = stamps.diff().dt.total_seconds().to_numpy()
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if positive.size == 0:
        return np.zeros(len(stamps), dtype=np.int64)
    modal = float(pd.Series(positive).mode().iloc[0])
    breaks = np.zeros(len(stamps), dtype=np.int64)
    breaks[1:] = (differences[1:] > gap_multiplier * modal).astype(np.int64)
    return np.cumsum(breaks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/weather.csv"))
    parser.add_argument("--target", default="OT")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--candidate-lags", default="0,1,2,3,6,12,24,48,95")
    parser.add_argument("--target-lags", default="0,1,2,6,12,24,48,95")
    parser.add_argument("--screening-horizon", type=int, default=96)
    parser.add_argument("--max-samples", type=int, default=36000)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--hac-bandwidth", type=int, default=95)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--minimum-effect", type=float, default=0.015)
    parser.add_argument("--minimum-stability", type=float, default=0.60)
    parser.add_argument("--maximum-pairs", type=int, default=11)
    parser.add_argument("--maximum-lags-per-feature", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--missing-sentinel", type=float, default=-9999.0, help="Known Weather missing-value sentinel used only for train-only discovery cleaning.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gcm/weather_gcm_manifest.json"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    if "date" not in frame or args.target not in frame:
        raise ValueError("Weather CSV must contain 'date' and target column")
    train_size = int(len(frame) * args.train_ratio)
    train = frame.iloc[:train_size].copy()
    feature_names = [c for c in train.columns if c not in {"date", args.target}]

    # The supplied Weather benchmark uses -9999 as a missing-sensor sentinel in
    # a small number of training rows (including OT).  Discovery must not treat
    # that code as a physical measurement.  Replace the known sentinel and use
    # strictly backward-looking forward fill *inside the training partition only*.
    # The downstream forecasting benchmark file itself is left untouched so the
    # published baseline protocol remains comparable.
    discovery_columns = [*feature_names, args.target]
    sentinel_counts = {}
    for column in discovery_columns:
        mask = np.isclose(
            train[column].to_numpy(dtype=np.float64),
            float(args.missing_sentinel),
            rtol=0.0,
            atol=1e-12,
        )
        sentinel_counts[column] = int(mask.sum())
        if mask.any():
            train.loc[mask, column] = np.nan
    train[discovery_columns] = train[discovery_columns].ffill()
    remaining_missing = train[discovery_columns].isna().sum()
    if int(remaining_missing.sum()) > 0:
        missing = remaining_missing[remaining_missing > 0].to_dict()
        raise ValueError(
            "Leading/unresolved missing values remain after time-valid forward fill: "
            f"{missing}"
        )

    manifest = discover_conditionally_relevant_lagged_drivers(
        features=train[feature_names].to_numpy(dtype=np.float64),
        target=train[args.target].to_numpy(dtype=np.float64),
        regimes=np.zeros(len(train), dtype=np.int64),
        segments=segments_from_time(train["date"]),
        feature_names=feature_names,
        candidate_lags=parse_ints(args.candidate_lags),
        conditioning_target_lags=parse_ints(args.target_lags),
        max_samples=args.max_samples,
        n_blocks=args.n_blocks,
        ridge_alpha=args.ridge_alpha,
        residual_learner="ridge",
        hac_bandwidth=args.hac_bandwidth,
        fdr_level=args.fdr,
        minimum_effect=args.minimum_effect,
        minimum_stability=args.minimum_stability,
        minimum_pairs=0,
        maximum_pairs=args.maximum_pairs,
        maximum_lags_per_feature=args.maximum_lags_per_feature,
        minimum_regime_samples=200,
        screening_horizon=args.screening_horizon,
        source_path=args.data,
        train_start=str(train.iloc[0]["date"]),
        train_end=str(train.iloc[-1]["date"]),
        random_state=args.seed,
    )
    manifest["data_quality"] = {
        "known_missing_sentinel": float(args.missing_sentinel),
        "sentinel_counts_in_training": {k: v for k, v in sentinel_counts.items() if v > 0},
        "discovery_imputation": "strictly_backward_forward_fill_within_training_partition",
        "forecast_benchmark_file_modified": False,
    }
    manifest["tdn_contract"] = {
        "downstream_driver_input": "full_observed_history_of_selected_variables",
        "selected_lag_role": "discovery_evidence_not_mandatory_input_shift",
        "forecast_future_unknown_driver_values": "never_used",
        "screening_partition": "chronological_training_only",
        "conditioning_semantics": "target_history_plus_all_other_observed_covariates_at_same_candidate_lag",
    }
    save_manifest(manifest, args.output)
    selected = [
        {
            "feature": record["feature"],
            "lag": record["lag"],
            "effect": record["effect"],
            "q_value": record["q_value"],
            "stability": record["stability"],
        }
        for record in manifest["selected_pairs"]
    ]
    print(json.dumps({"selected_count": len(selected), "selected_pairs": selected}, indent=2))
    print(f"Manifest written to {args.output}")


if __name__ == "__main__":
    main()
