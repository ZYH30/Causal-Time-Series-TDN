"""TDN journal contract for train-only lagged GCM driver screening.

The module adapts the latest temporal GCM implementation from the supplied
SF-CDI reference code to the TDN forecasting setting.  It uses training data
only, forward-blocked residualization, HAC/Newey-West uncertainty, FDR control,
effect-size and temporal-stability gates, and a bounded variable-lag manifest.

The paper-facing TDN wrapper additionally performs candidate-specific conditioning
on target-history lags and all other observed covariates at the same tested
historical slice. The TDN paper uses a deliberately conservative interpretation:
selected pairs are *lagged causal-driver candidates under the observed-variable
and testing assumptions*. The manifest is a governed forecasting input view; it
is not a complete causal graph and does not estimate intervention effects.  Lags are
discovery evidence by default; the downstream TDN consumes the selected
variables' complete observed histories unless an experiment explicitly opts
into lag alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import json
import math

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


@dataclass(frozen=True)
class Candidate:
    feature_index: int
    feature: str
    lag: int


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    m = len(p_values)
    if m == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """Return BY-adjusted p-values, valid under arbitrary test dependence."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    m = len(p_values)
    if m == 0:
        return p_values.copy()
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * m * harmonic / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _sha256_file(path: str | Path, chunk_size: int = 2**20) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_content_sha256(manifest: dict) -> str:
    """Hash scientific manifest content while excluding run-time metadata."""

    content = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_utc", "manifest_sha256"}
    }
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(canonical).hexdigest()


def _residual_estimator(
    learner: str,
    ridge_alpha: float,
    random_state: int,
    options: dict | None,
):
    options = dict(options or {})
    if learner == "ridge":
        return Ridge(alpha=ridge_alpha, fit_intercept=True)
    if learner == "random_forest":
        defaults = {
            "n_estimators": 32,
            "max_depth": 8,
            "min_samples_leaf": 32,
            "max_features": 1.0,
            "n_jobs": -1,
        }
        defaults.update(options)
        return RandomForestRegressor(random_state=random_state, **defaults)
    if learner == "shallow_mlp":
        defaults = {
            "hidden_layer_sizes": (32,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-3,
            "batch_size": 512,
            "learning_rate_init": 1e-3,
            "max_iter": 160,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 12,
        }
        defaults.update(options)
        return MLPRegressor(random_state=random_state, **defaults)
    if learner == "spline_ridge":
        n_knots = int(options.pop("n_knots", 5))
        degree = int(options.pop("degree", 3))
        if options:
            raise ValueError(f"Unknown spline_ridge options: {sorted(options)}")
        return make_pipeline(
            SplineTransformer(
                n_knots=n_knots,
                degree=degree,
                include_bias=False,
            ),
            Ridge(alpha=ridge_alpha, fit_intercept=True),
        )
    raise ValueError(f"Unknown residual learner: {learner}")


def _residuals_forward(
    conditioning: np.ndarray,
    outcomes: np.ndarray,
    n_blocks: int,
    ridge_alpha: float,
    groups: np.ndarray | None = None,
    learner: str = "ridge",
    learner_options: dict | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residualize outcomes using prior time blocks only.

    Block zero is a warm-up block and is never evaluated.  Every reported
    residual is therefore produced by a model fitted strictly on earlier rows.
    """

    n = len(conditioning)
    if n_blocks < 3:
        raise ValueError("At least three blocks are required")
    if n < n_blocks * 20:
        raise ValueError("Too few observations for blocked cross-fitting")
    if groups is None:
        group_rows = [np.arange(n, dtype=np.int64)]
    else:
        groups = np.asarray(groups)
        all_group_rows = [
            np.flatnonzero(groups == group) for group in np.unique(groups)
        ]
        # Industrial logs often contain many short fragments around acquisition
        # gaps.  Such fragments cannot support a five-block forward fit and are
        # excluded from discovery instead of forcing the entire dataset back to
        # a global cross-fit that could bridge physical discontinuities.
        group_rows = [rows for rows in all_group_rows if len(rows) >= n_blocks * 20]
        if not group_rows:
            raise ValueError(
                "No cross-fit group has at least 20 rows per block"
            )
    residual_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    for fold in range(1, n_blocks):
        train_indices = []
        evaluation_indices = []
        for rows in group_rows:
            boundaries = np.linspace(0, len(rows), n_blocks + 1, dtype=np.int64)
            train_indices.append(rows[: boundaries[fold]])
            evaluation_indices.append(rows[boundaries[fold] : boundaries[fold + 1]])
        train_index = np.sort(np.concatenate(train_indices))
        evaluation_index = np.sort(np.concatenate(evaluation_indices))
        scaler = StandardScaler().fit(conditioning[train_index])
        z_train = scaler.transform(conditioning[train_index])
        z_eval = scaler.transform(conditioning[evaluation_index])
        outcome_scaler = StandardScaler().fit(outcomes[train_index])
        outcomes_train = outcome_scaler.transform(outcomes[train_index])
        estimator = _residual_estimator(
            learner,
            ridge_alpha=ridge_alpha,
            random_state=random_state + fold,
            options=learner_options,
        )
        estimator.fit(z_train, outcomes_train)
        predicted = outcome_scaler.inverse_transform(estimator.predict(z_eval))
        residual_parts.append(outcomes[evaluation_index] - predicted)
        index_parts.append(evaluation_index)
        fold_parts.append(np.full(len(evaluation_index), fold, dtype=np.int64))
    return (
        np.concatenate(residual_parts, axis=0),
        np.concatenate(index_parts),
        np.concatenate(fold_parts),
    )


def _ridge_residuals_forward(
    conditioning: np.ndarray,
    outcomes: np.ndarray,
    n_blocks: int,
    ridge_alpha: float,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for the default Ridge residual learner."""

    return _residuals_forward(
        conditioning=conditioning,
        outcomes=outcomes,
        n_blocks=n_blocks,
        ridge_alpha=ridge_alpha,
        groups=groups,
        learner="ridge",
    )


def _selected_block_permutation_p_values(
    candidate_residuals: np.ndarray,
    target_residuals: np.ndarray,
    groups: np.ndarray,
    selected_indices: Sequence[int],
    permutations: int,
    block_length: int,
    random_state: int,
) -> dict[int, float]:
    """Audit selected residual products using within-group block permutations."""

    if permutations < 1 or not selected_indices:
        return {}
    selected_indices = [int(index) for index in selected_indices]
    selected_residuals = candidate_residuals[:, selected_indices]
    observed = np.abs(np.mean(selected_residuals * target_residuals[:, None], axis=0))
    exceedances = np.zeros(len(selected_indices), dtype=np.int64)
    generator = np.random.default_rng(random_state)
    group_rows = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    for _ in range(permutations):
        permuted = target_residuals.copy()
        for rows in group_rows:
            blocks = [
                rows[start : start + block_length]
                for start in range(0, len(rows), block_length)
            ]
            if len(blocks) > 1:
                order = generator.permutation(len(blocks))
                source = np.concatenate([blocks[index] for index in order])
                permuted[rows] = target_residuals[source]
            elif len(rows) > 1:
                shift = int(generator.integers(1, len(rows)))
                permuted[rows] = np.roll(target_residuals[rows], shift)
        permuted_effect = np.abs(
            np.mean(selected_residuals * permuted[:, None], axis=0)
        )
        exceedances += permuted_effect >= observed
    p_values = (exceedances + 1.0) / (permutations + 1.0)
    return {
        index: float(p_value)
        for index, p_value in zip(selected_indices, p_values)
    }


def _hac_mean_test(
    values: np.ndarray,
    groups: np.ndarray,
    bandwidth: int,
) -> tuple[float, float, float]:
    """Two-sided HAC test that the ordered product mean equals zero."""

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    n = len(values)
    mean = float(np.mean(values))
    centered = values - mean
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for lag in range(1, min(bandwidth, n - 1) + 1):
        same_group = groups[lag:] == groups[:-lag]
        if not np.any(same_group):
            continue
        covariance = float(
            np.dot(centered[lag:][same_group], centered[:-lag][same_group]) / n
        )
        weight = 1.0 - lag / (bandwidth + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, gamma0 / max(n, 1), 1e-16)
    standard_error = math.sqrt(long_run_variance / n)
    statistic = mean / standard_error
    p_value = float(2.0 * norm.sf(abs(statistic)))
    return statistic, p_value, standard_error


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _valid_origins(
    segments: np.ndarray,
    maximum_lag: int,
    future_horizon: int = 1,
) -> np.ndarray:
    origins = np.arange(maximum_lag, len(segments) - future_horizon, dtype=np.int64)
    valid = segments[origins - maximum_lag] == segments[origins + future_horizon]
    return origins[valid]


def discover_lagged_drivers(
    features: np.ndarray,
    target: np.ndarray,
    regimes: np.ndarray,
    segments: np.ndarray,
    feature_names: Sequence[str],
    candidate_lags: Sequence[int],
    conditioning_target_lags: Sequence[int] = (0, 1, 2, 6, 12),
    max_samples: int = 50_000,
    n_blocks: int = 5,
    ridge_alpha: float = 1.0,
    residual_learner: str = "ridge",
    residual_learner_options: dict | None = None,
    hac_bandwidth: int = 12,
    fdr_level: float = 0.05,
    minimum_effect: float = 0.015,
    minimum_stability: float = 0.60,
    minimum_pairs: int = 0,
    maximum_pairs: int = 12,
    maximum_lags_per_feature: int = 2,
    minimum_regime_samples: int = 200,
    crossfit_within_segments: bool = False,
    screening_horizon: int = 1,
    source_path: str | Path | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    block_permutations: int = 0,
    permutation_block_length: int | None = None,
    random_state: int = 0,
) -> dict:
    """Discover a bounded, auditable set of lagged predictive-driver pairs."""

    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    regimes = np.asarray(regimes, dtype=np.int64)
    segments = np.asarray(segments, dtype=np.int64)
    candidate_lags = sorted({int(lag) for lag in candidate_lags})
    conditioning_target_lags = sorted({int(lag) for lag in conditioning_target_lags})
    if not candidate_lags or min(candidate_lags) < 0:
        raise ValueError("candidate_lags must contain non-negative integers")
    if features.shape != (len(target), len(feature_names)):
        raise ValueError("features, target, and feature_names have inconsistent shapes")

    maximum_lag = max([*candidate_lags, *conditioning_target_lags])
    screening_horizon = int(screening_horizon)
    if screening_horizon < 1:
        raise ValueError("screening_horizon must be positive")
    origins = _valid_origins(segments, maximum_lag, screening_horizon)
    if len(origins) > max_samples:
        positions = np.linspace(0, len(origins) - 1, max_samples, dtype=np.int64)
        origins = origins[positions]

    candidates = [
        Candidate(feature_index=j, feature=name, lag=lag)
        for j, name in enumerate(feature_names)
        for lag in candidate_lags
    ]
    candidate_matrix = np.column_stack(
        [features[origins - candidate.lag, candidate.feature_index] for candidate in candidates]
    )
    target_future = np.mean(
        np.column_stack(
            [target[origins + step] for step in range(1, screening_horizon + 1)]
        ),
        axis=1,
        keepdims=True,
    )
    target_history = np.column_stack(
        [target[origins - lag] for lag in conditioning_target_lags]
    )
    unique_regimes = np.unique(regimes)
    regime_design = np.column_stack(
        [(regimes[origins] == regime).astype(np.float64) for regime in unique_regimes]
    )
    conditioning = np.column_stack([target_history, regime_design])
    outcomes = np.column_stack([candidate_matrix, target_future])

    crossfit_groups = segments[origins] if crossfit_within_segments else None
    residuals, evaluated_rows, fold_ids = _residuals_forward(
        conditioning=conditioning,
        outcomes=outcomes,
        n_blocks=n_blocks,
        ridge_alpha=ridge_alpha,
        groups=crossfit_groups,
        learner=residual_learner,
        learner_options=residual_learner_options,
        random_state=random_state,
    )
    candidate_residuals = residuals[:, :-1]
    target_residuals = residuals[:, -1]
    evaluated_origins = origins[evaluated_rows]
    evaluated_regimes = regimes[evaluated_origins]
    # A group change prevents HAC covariance terms spanning a fold, regime, or
    # physical discontinuity.
    hac_groups = (
        fold_ids.astype(np.int64) * (len(unique_regimes) + 1) * (segments.max() + 2)
        + evaluated_regimes * (segments.max() + 2)
        + segments[evaluated_origins]
    )

    records: list[dict] = []
    p_values: list[float] = []
    for column, candidate in enumerate(candidates):
        driver_residual = candidate_residuals[:, column]
        product = driver_residual * target_residuals
        statistic, p_value, standard_error = _hac_mean_test(
            product, hac_groups, bandwidth=hac_bandwidth
        )
        effect = _safe_corr(driver_residual, target_residuals)
        block_effects = [
            _safe_corr(driver_residual[fold_ids == fold], target_residuals[fold_ids == fold])
            for fold in np.unique(fold_ids)
        ]
        effect_sign = 1.0 if effect >= 0 else -1.0
        stable_blocks = [
            abs(value) >= minimum_effect and np.sign(value) == effect_sign
            for value in block_effects
        ]
        stability = float(np.mean(stable_blocks)) if stable_blocks else 0.0
        regime_effects: dict[str, dict] = {}
        strong_regimes: list[int] = []
        for regime in unique_regimes:
            mask = evaluated_regimes == regime
            count = int(mask.sum())
            regime_effect = _safe_corr(driver_residual[mask], target_residuals[mask])
            regime_effects[str(int(regime))] = {
                "n": count,
                "residual_correlation": regime_effect,
            }
            if (
                count >= minimum_regime_samples
                and abs(regime_effect) >= minimum_effect
                and np.sign(regime_effect) == effect_sign
            ):
                strong_regimes.append(int(regime))
        records.append(
            {
                "feature": candidate.feature,
                "feature_index": candidate.feature_index,
                "lag": candidate.lag,
                "effect": effect,
                "hac_statistic": statistic,
                "hac_standard_error": standard_error,
                "p_value": p_value,
                "block_effects": block_effects,
                "stability": stability,
                "regime_effects": regime_effects,
                "strong_regimes": strong_regimes,
            }
        )
        p_values.append(p_value)

    q_values = benjamini_hochberg(np.asarray(p_values))
    q_values_by = benjamini_yekutieli(np.asarray(p_values))
    for record, q_value, q_value_by in zip(records, q_values, q_values_by):
        record["q_value"] = float(q_value)
        record["q_value_by"] = float(q_value_by)
        record["passes_inference"] = bool(
            q_value <= fdr_level
            and abs(record["effect"]) >= minimum_effect
            and record["stability"] >= minimum_stability
        )
        record["passes_by_inference"] = bool(
            q_value_by <= fdr_level
            and abs(record["effect"]) >= minimum_effect
            and record["stability"] >= minimum_stability
        )

    ranking = sorted(
        range(len(records)),
        key=lambda index: (
            not records[index]["passes_inference"],
            -records[index]["stability"],
            -abs(records[index]["effect"]),
            records[index]["q_value"],
        ),
    )
    selected_indices: list[int] = []
    feature_counts: dict[str, int] = {}
    for index in ranking:
        record = records[index]
        feature_count = feature_counts.get(record["feature"], 0)
        if feature_count >= maximum_lags_per_feature:
            continue
        if len(selected_indices) >= maximum_pairs:
            break
        if record["passes_inference"] or len(selected_indices) < minimum_pairs:
            selected_indices.append(index)
            feature_counts[record["feature"]] = feature_count + 1

    selected: list[dict] = []
    for index in selected_indices:
        record = dict(records[index])
        record["selection_basis"] = (
            "fdr_effect_stability"
            if record["passes_inference"]
            else "screening_fallback"
        )
        strong = record["strong_regimes"]
        # A pair is a stable core only when supported in at least two observed
        # regimes.  Otherwise its deployment mask is restricted to supported
        # regimes.
        if record["passes_inference"] and len(strong) >= 2:
            record["governance"] = "stable_core"
            record["available_regimes"] = [int(value) for value in unique_regimes]
        elif record["passes_inference"] and strong:
            record["governance"] = "regime_specific"
            record["available_regimes"] = strong
        else:
            record["governance"] = "screening_only"
            record["available_regimes"] = [int(value) for value in unique_regimes]
        selected.append(record)

    permutation_p_values = _selected_block_permutation_p_values(
        candidate_residuals,
        target_residuals,
        hac_groups,
        selected_indices,
        permutations=int(block_permutations),
        block_length=int(
            permutation_block_length
            if permutation_block_length is not None
            else hac_bandwidth + 1
        ),
        random_state=int(random_state),
    )
    for index, selected_record in zip(selected_indices, selected):
        permutation_p_value = permutation_p_values.get(index)
        selected_record["block_permutation_p_value"] = permutation_p_value
        records[index]["block_permutation_p_value"] = permutation_p_value

    source_hash = _sha256_file(source_path) if source_path else None
    crossfit_group_counts = (
        np.unique(crossfit_groups, return_counts=True)[1]
        if crossfit_groups is not None
        else np.asarray([len(origins)], dtype=np.int64)
    )
    minimum_crossfit_group_rows = n_blocks * 20
    retained_crossfit_groups = crossfit_group_counts >= minimum_crossfit_group_rows
    manifest = {
        "schema_version": "1.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Selected variable-lag pairs are treated as lagged causal-driver candidates "
            "under the observed-variable boundary and the recorded temporal testing "
            "assumptions. The manifest is a forecasting input-governance object, not a "
            "complete causal graph and not an intervention-effect estimate."
        ),
        "source": {"path": str(source_path) if source_path else None, "sha256": source_hash},
        "training_interval": {"start": train_start, "end": train_end},
        "sample_counts": {
            "eligible": int(
                len(_valid_origins(segments, maximum_lag, screening_horizon))
            ),
            "screened": int(len(origins)),
            "crossfit_evaluated": int(len(evaluated_rows)),
            "crossfit_groups_screened": int(len(crossfit_group_counts)),
            "crossfit_groups_retained": int(np.sum(retained_crossfit_groups)),
            "crossfit_rows_excluded_short_groups": int(
                np.sum(crossfit_group_counts[~retained_crossfit_groups])
            ),
        },
        "configuration": {
            "candidate_lags": candidate_lags,
            "conditioning_target_lags": conditioning_target_lags,
            "n_blocks": n_blocks,
            "ridge_alpha": ridge_alpha,
            "residual_learner": residual_learner,
            "residual_learner_options": residual_learner_options or {},
            "hac_bandwidth": hac_bandwidth,
            "fdr_level": fdr_level,
            "minimum_effect": minimum_effect,
            "minimum_stability": minimum_stability,
            "minimum_pairs": minimum_pairs,
            "maximum_pairs": maximum_pairs,
            "maximum_lags_per_feature": maximum_lags_per_feature,
            "crossfit_within_segments": crossfit_within_segments,
            "minimum_crossfit_rows_per_group": minimum_crossfit_group_rows,
            "screening_horizon": screening_horizon,
            "screening_target": "mean_future_target_over_horizon",
            "selection_ranking": "inference_pass_then_stability_then_absolute_effect",
            "multiple_testing": {
                "primary": "Benjamini-Hochberg FDR",
                "sensitivity": (
                    "Benjamini-Yekutieli FDR under arbitrary dependence"
                ),
            },
            "block_permutation_audit": {
                "permutations": int(block_permutations),
                "block_length": int(
                    permutation_block_length
                    if permutation_block_length is not None
                    else hac_bandwidth + 1
                ),
                "scope": "selected_pairs_only",
            },
        },
        "regime_ids": [int(value) for value in unique_regimes],
        "selected_pairs": selected,
        "all_candidates": records,
    }
    manifest["manifest_sha256"] = manifest_content_sha256(manifest)
    return manifest


def save_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load_manifest(path: str | Path) -> dict:
    """Load and minimally validate a saved driver manifest."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "selected_pairs" not in manifest or "configuration" not in manifest:
        raise ValueError(f"Invalid TDN GCM manifest: {path}")
    return manifest


def discover_conditionally_relevant_lagged_drivers(
    features: np.ndarray,
    target: np.ndarray,
    regimes: np.ndarray,
    segments: np.ndarray,
    feature_names: Sequence[str],
    candidate_lags: Sequence[int],
    conditioning_target_lags: Sequence[int] = (0, 1, 2, 6, 12),
    max_samples: int = 20_000,
    n_blocks: int = 5,
    ridge_alpha: float = 1.0,
    residual_learner: str = "ridge",
    residual_learner_options: dict | None = None,
    hac_bandwidth: int = 12,
    fdr_level: float = 0.05,
    minimum_effect: float = 0.015,
    minimum_stability: float = 0.60,
    minimum_pairs: int = 0,
    maximum_pairs: int = 11,
    maximum_lags_per_feature: int = 1,
    minimum_regime_samples: int = 200,
    screening_horizon: int = 96,
    source_path: str | Path | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    random_state: int = 0,
) -> dict:
    """TDN-specific lagged GCM with observed-covariate conditioning.

    This is the journal TDN contract, not the stronger causal interpretation used
    by the supplied mechanical reference.  For each candidate ``X_j(t-l)``, both
    the candidate and future-target screening response are residualized against:

    * prespecified target-history lags;
    * all *other* observed covariates at the same historical slice ``t-l``;
    * optional regime indicators.

    The residual regressions are fit on earlier chronological blocks and predict
    the next block.  GCM/HAC is then applied to the out-of-fold residual product.
    Thus a retained pair represents *conditional relevance after accounting for
    the remaining observed covariates and target inertia*, matching TDN's stated
    observed-variable-boundary semantics.  It is not a direct-parent proof.
    """
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    regimes = np.asarray(regimes, dtype=np.int64)
    segments = np.asarray(segments, dtype=np.int64)
    feature_names = list(feature_names)
    candidate_lags = sorted({int(lag) for lag in candidate_lags})
    conditioning_target_lags = sorted({int(lag) for lag in conditioning_target_lags})
    if not candidate_lags or min(candidate_lags) < 0:
        raise ValueError("candidate_lags must contain non-negative integers")
    if features.shape != (len(target), len(feature_names)):
        raise ValueError("features, target, and feature_names have inconsistent shapes")
    if features.shape[1] < 2:
        raise ValueError("Observed-covariate conditioning requires at least two features")

    maximum_lag = max([*candidate_lags, *conditioning_target_lags])
    screening_horizon = int(screening_horizon)
    if screening_horizon < 1:
        raise ValueError("screening_horizon must be positive")
    origins = _valid_origins(segments, maximum_lag, screening_horizon)
    if len(origins) > max_samples:
        positions = np.linspace(0, len(origins) - 1, max_samples, dtype=np.int64)
        origins = origins[positions]

    target_future = np.mean(
        np.column_stack(
            [target[origins + step] for step in range(1, screening_horizon + 1)]
        ),
        axis=1,
    )
    target_history = np.column_stack(
        [target[origins - lag] for lag in conditioning_target_lags]
    )
    unique_regimes = np.unique(regimes)
    regime_design = np.column_stack(
        [(regimes[origins] == regime).astype(np.float64) for regime in unique_regimes]
    )

    records: list[dict] = []
    p_values: list[float] = []
    crossfit_rows: int | None = None

    # Candidate-specific conditioning is essential here: including X_j itself in
    # the control set would residualize the tested variable away.  At each lag we
    # use all other physical covariates measured at that same historical slice.
    for lag_index, lag in enumerate(candidate_lags):
        lagged_features = features[origins - lag, :]
        for feature_index, feature in enumerate(feature_names):
            candidate = lagged_features[:, feature_index]
            other_covariates = np.delete(lagged_features, feature_index, axis=1)
            conditioning = np.column_stack(
                [target_history, other_covariates, regime_design]
            )
            outcomes = np.column_stack([candidate, target_future])
            residuals, evaluated_rows, fold_ids = _residuals_forward(
                conditioning=conditioning,
                outcomes=outcomes,
                n_blocks=n_blocks,
                ridge_alpha=ridge_alpha,
                groups=None,
                learner=residual_learner,
                learner_options=residual_learner_options,
                random_state=random_state + lag_index * len(feature_names) + feature_index,
            )
            if crossfit_rows is None:
                crossfit_rows = int(len(evaluated_rows))
            candidate_residual = residuals[:, 0]
            target_residual = residuals[:, 1]
            evaluated_origins = origins[evaluated_rows]
            evaluated_regimes = regimes[evaluated_origins]
            # Fold/regime/physical-segment changes form hard HAC boundaries.
            hac_groups = (
                fold_ids.astype(np.int64)
                * (len(unique_regimes) + 1)
                * (segments.max() + 2)
                + evaluated_regimes * (segments.max() + 2)
                + segments[evaluated_origins]
            )
            product = candidate_residual * target_residual
            statistic, p_value, standard_error = _hac_mean_test(
                product, hac_groups, bandwidth=hac_bandwidth
            )
            effect = _safe_corr(candidate_residual, target_residual)
            block_effects = [
                _safe_corr(
                    candidate_residual[fold_ids == fold],
                    target_residual[fold_ids == fold],
                )
                for fold in np.unique(fold_ids)
            ]
            effect_sign = 1.0 if effect >= 0 else -1.0
            stable_blocks = [
                abs(value) >= minimum_effect and np.sign(value) == effect_sign
                for value in block_effects
            ]
            stability = float(np.mean(stable_blocks)) if stable_blocks else 0.0
            regime_effects: dict[str, dict] = {}
            strong_regimes: list[int] = []
            for regime in unique_regimes:
                mask = evaluated_regimes == regime
                count = int(mask.sum())
                regime_effect = _safe_corr(
                    candidate_residual[mask], target_residual[mask]
                )
                regime_effects[str(int(regime))] = {
                    "n": count,
                    "residual_correlation": regime_effect,
                }
                if (
                    count >= minimum_regime_samples
                    and abs(regime_effect) >= minimum_effect
                    and np.sign(regime_effect) == effect_sign
                ):
                    strong_regimes.append(int(regime))
            records.append(
                {
                    "feature": feature,
                    "feature_index": feature_index,
                    "lag": int(lag),
                    "effect": effect,
                    "hac_statistic": statistic,
                    "hac_standard_error": standard_error,
                    "p_value": p_value,
                    "block_effects": block_effects,
                    "stability": stability,
                    "regime_effects": regime_effects,
                    "strong_regimes": strong_regimes,
                }
            )
            p_values.append(p_value)

    q_values = benjamini_hochberg(np.asarray(p_values))
    q_values_by = benjamini_yekutieli(np.asarray(p_values))
    for record, q_value, q_value_by in zip(records, q_values, q_values_by):
        record["q_value"] = float(q_value)
        record["q_value_by"] = float(q_value_by)
        record["passes_inference"] = bool(
            q_value <= fdr_level
            and abs(record["effect"]) >= minimum_effect
            and record["stability"] >= minimum_stability
        )
        record["passes_by_inference"] = bool(
            q_value_by <= fdr_level
            and abs(record["effect"]) >= minimum_effect
            and record["stability"] >= minimum_stability
        )

    ranking = sorted(
        range(len(records)),
        key=lambda index: (
            not records[index]["passes_inference"],
            -records[index]["stability"],
            -abs(records[index]["effect"]),
            records[index]["q_value"],
        ),
    )
    selected_indices: list[int] = []
    feature_counts: dict[str, int] = {}
    for index in ranking:
        record = records[index]
        count = feature_counts.get(record["feature"], 0)
        if count >= maximum_lags_per_feature:
            continue
        if len(selected_indices) >= maximum_pairs:
            break
        if record["passes_inference"] or len(selected_indices) < minimum_pairs:
            selected_indices.append(index)
            feature_counts[record["feature"]] = count + 1

    selected: list[dict] = []
    for index in selected_indices:
        record = dict(records[index])
        record["selection_basis"] = (
            "fdr_effect_stability"
            if record["passes_inference"]
            else "screening_fallback"
        )
        strong = record["strong_regimes"]
        if record["passes_inference"] and len(strong) >= 2:
            record["governance"] = "stable_core"
            record["available_regimes"] = [int(value) for value in unique_regimes]
        elif record["passes_inference"] and strong:
            record["governance"] = "regime_specific"
            record["available_regimes"] = strong
        else:
            record["governance"] = "screening_only"
            record["available_regimes"] = [int(value) for value in unique_regimes]
        record["block_permutation_p_value"] = None
        selected.append(record)

    source_hash = _sha256_file(source_path) if source_path else None
    manifest = {
        "schema_version": "tdn-gcm-2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Selected variable-lag pairs are lagged causal-driver candidates under "
            "the observed-variable and temporal-testing assumptions. Each candidate "
            "remains conditionally relevant after accounting for target-history lags "
            "and the other observed covariates at the tested historical slice. The "
            "manifest is not a complete causal graph, direct-parent proof, or "
            "intervention-effect estimate."
        ),
        "source": {"path": str(source_path) if source_path else None, "sha256": source_hash},
        "training_interval": {"start": train_start, "end": train_end},
        "sample_counts": {
            "eligible": int(
                len(_valid_origins(segments, maximum_lag, screening_horizon))
            ),
            "screened": int(len(origins)),
            "crossfit_evaluated_per_test": int(crossfit_rows or 0),
            "candidate_tests": int(len(records)),
        },
        "configuration": {
            "candidate_lags": candidate_lags,
            "conditioning_target_lags": conditioning_target_lags,
            "conditioning_observed_covariates": (
                "all_other_features_at_same_candidate_lag"
            ),
            "n_blocks": n_blocks,
            "ridge_alpha": ridge_alpha,
            "residual_learner": residual_learner,
            "residual_learner_options": residual_learner_options or {},
            "hac_bandwidth": hac_bandwidth,
            "fdr_level": fdr_level,
            "minimum_effect": minimum_effect,
            "minimum_stability": minimum_stability,
            "minimum_pairs": minimum_pairs,
            "maximum_pairs": maximum_pairs,
            "maximum_lags_per_feature": maximum_lags_per_feature,
            "screening_horizon": screening_horizon,
            "screening_target": "mean_future_target_over_horizon",
            "selection_ranking": "inference_pass_then_stability_then_absolute_effect",
            "multiple_testing": {
                "primary": "Benjamini-Hochberg FDR",
                "sensitivity": "Benjamini-Yekutieli FDR under arbitrary dependence",
            },
        },
        "regime_ids": [int(value) for value in unique_regimes],
        "selected_pairs": selected,
        "all_candidates": records,
    }
    manifest["manifest_sha256"] = manifest_content_sha256(manifest)
    return manifest
