"""Cardinality-matched Weather selectors used only as a controlled ablation.

All selectors see the same chronological training partition and the same
future-mean screening response.  TDN is unchanged; only its admitted driver
variables change.  This experiment asks whether GCM's benefit can be explained
solely by reducing the number of input variables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from gcm.contract import load_manifest


def _future_mean_response(target: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    return np.mean(
        np.column_stack([target[origins + step] for step in range(1, horizon + 1)]),
        axis=1,
    )


def _safe_abs_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(left, right)[0, 1]))


def _best_lag_scores(
    train: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    candidate_lags: list[int],
    screening_horizon: int,
    method: str,
    seed: int,
    max_mi_samples: int = 20000,
) -> list[dict]:
    maximum_lag = max(candidate_lags)
    origins = np.arange(maximum_lag, len(train) - screening_horizon, dtype=np.int64)
    target = train[target_name].to_numpy(dtype=np.float64)
    response = _future_mean_response(target, origins, screening_horizon)
    records: list[dict] = []
    for feature in feature_names:
        values = train[feature].to_numpy(dtype=np.float64)
        lag_scores: list[tuple[int, float]] = []
        for lag in candidate_lags:
            candidate = values[origins - lag]
            if method == "correlation":
                score = _safe_abs_corr(candidate, response)
            elif method == "mutual_information":
                if len(origins) > max_mi_samples:
                    positions = np.linspace(0, len(origins) - 1, max_mi_samples, dtype=np.int64)
                    x = candidate[positions, None]
                    y = response[positions]
                else:
                    x = candidate[:, None]
                    y = response
                score = float(
                    mutual_info_regression(
                        x,
                        y,
                        random_state=seed,
                        n_neighbors=3,
                    )[0]
                )
            else:
                raise ValueError(method)
            lag_scores.append((int(lag), score))
        best_lag, best_score = max(lag_scores, key=lambda item: item[1])
        records.append(
            {
                "feature": feature,
                "best_lag": best_lag,
                "score": float(best_score),
                "lag_scores": [{"lag": lag, "score": score} for lag, score in lag_scores],
            }
        )
    return sorted(records, key=lambda record: -record["score"])


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_selector_manifests(
    data_path: str | Path,
    gcm_manifest_path: str | Path,
    output_dir: str | Path,
    target: str = "OT",
    train_ratio: float = 0.70,
    random_repeats: int = 5,
    seed: int = 2026,
    missing_sentinel: float = -9999.0,
) -> dict[str, Path]:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    frame = pd.read_csv(data_path)
    n_train = int(len(frame) * train_ratio)
    train = frame.iloc[:n_train].copy()
    feature_names = [c for c in train.columns if c not in {"date", target}]
    selector_columns = [*feature_names, target]
    for column in selector_columns:
        mask = np.isclose(
            train[column].to_numpy(dtype=np.float64),
            float(missing_sentinel),
            rtol=0.0,
            atol=1e-12,
        )
        if mask.any():
            train.loc[mask, column] = np.nan
    train[selector_columns] = train[selector_columns].ffill()
    if train[selector_columns].isna().any().any():
        raise ValueError("Unresolved leading missing values after train-only forward fill")

    gcm = load_manifest(gcm_manifest_path)
    gcm_features = []
    for pair in gcm["selected_pairs"]:
        if pair["feature"] not in gcm_features:
            gcm_features.append(pair["feature"])
    k = len(gcm_features)
    if k == 0:
        raise ValueError("GCM manifest selected no variables")
    configuration = gcm["configuration"]
    candidate_lags = [int(v) for v in configuration["candidate_lags"]]
    screening_horizon = int(configuration["screening_horizon"])

    common = {
        "schema_version": "tdn-selector-1.0",
        "data": str(data_path),
        "target": target,
        "train_ratio": train_ratio,
        "cardinality": k,
        "candidate_lags": candidate_lags,
        "screening_horizon": screening_horizon,
        "downstream_input": "full_observed_history_of_selected_variables",
        "data_quality": {
            "known_missing_sentinel": float(missing_sentinel),
            "selector_preprocessing": "strictly_backward_forward_fill_within_training_partition",
        },
    }

    paths: dict[str, Path] = {}
    gcm_path = output_dir / "selector_gcm.json"
    _write_manifest(
        gcm_path,
        {
            **common,
            "selector": "gcm",
            "selected_features": gcm_features,
            "source_manifest": str(gcm_manifest_path),
            "selected_evidence": [
                {"feature": p["feature"], "lag": p["lag"], "q_value": p["q_value"]}
                for p in gcm["selected_pairs"]
            ],
        },
    )
    paths["gcm"] = gcm_path

    for method, label in [
        ("correlation", "correlation"),
        ("mutual_information", "mutual_information"),
    ]:
        ranking = _best_lag_scores(
            train,
            feature_names,
            target,
            candidate_lags,
            screening_horizon,
            method=method,
            seed=seed,
        )
        path = output_dir / f"selector_{label}.json"
        _write_manifest(
            path,
            {
                **common,
                "selector": label,
                "selected_features": [record["feature"] for record in ranking[:k]],
                "ranking": ranking,
            },
        )
        paths[label] = path

    generator = np.random.default_rng(seed)
    for repeat in range(random_repeats):
        selected = generator.choice(feature_names, size=k, replace=False).tolist()
        path = output_dir / f"selector_random_{repeat:02d}.json"
        _write_manifest(
            path,
            {
                **common,
                "selector": "random",
                "repeat": repeat,
                "random_seed": seed,
                "selected_features": selected,
            },
        )
        paths[f"random_{repeat:02d}"] = path

    all_path = output_dir / "selector_all_variables.json"
    _write_manifest(
        all_path,
        {
            **common,
            "selector": "all_variables",
            "cardinality": len(feature_names),
            "selected_features": feature_names,
        },
    )
    paths["all_variables"] = all_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/weather.csv"))
    parser.add_argument(
        "--gcm-manifest",
        type=Path,
        default=Path("results/gcm/weather_gcm_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/selectors"),
    )
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--missing-sentinel", type=float, default=-9999.0)
    args = parser.parse_args()
    paths = build_selector_manifests(
        data_path=args.data,
        gcm_manifest_path=args.gcm_manifest,
        output_dir=args.output_dir,
        random_repeats=args.random_repeats,
        seed=args.seed,
        missing_sentinel=args.missing_sentinel,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
