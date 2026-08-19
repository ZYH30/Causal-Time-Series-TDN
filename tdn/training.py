"""Training contract for journal TDN.

The core design is a *decoupled alternating minimax* game.

Warm-up (utility initialization)
    Future-target loss may update driver + target/forecast parameters so the
    driver representation starts from a forecast-useful coordinate system.
    The adversary is fitted on the detached driver representation.

Alternating phase, each mini-batch
    MAX / utility-aware driver-adjustment step:
        update ONLY the driver encoder with one deliberately constructed
        objective: preserve future-target utility while making the
        target-history summary harder to reconstruct. Forecast and adversary
        parameters are frozen.

    MIN / forecast-and-probe step:
        minimize future forecasting error and target-history reconstruction
        error; the driver representation is detached. Update ONLY the target
        path / future decoder and the adversarial history decoder.

This prevents the driver encoder from being pulled by separate optimizers in
opposite directions while avoiding the V1.2 failure mode in which a pure
confusion step could erase future-relevant driver information.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import json
import random

import numpy as np
import torch
from torch import nn

from .model import TDNJournal
from .probes import evaluate_independent_probes


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def target_history_summary(y_history: torch.Tensor) -> torch.Tensor:
    """Default S_Y(H_Y): level, mean, variability, and end-to-end trend.

    ``y_history`` is already standardized by the train-only dataset scaler, so
    these four coordinates have compatible numerical scales while remaining
    nontrivial under the model's internal RevIN normalization.
    """
    last = y_history[:, -1, 0]
    mean = y_history[:, :, 0].mean(dim=1)
    std = y_history[:, :, 0].std(dim=1, unbiased=False)
    trend = y_history[:, -1, 0] - y_history[:, 0, 0]
    return torch.stack([last, mean, std, trend], dim=1)


def _set_requires_grad(parameters: Iterable[nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def _clear_gradients(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.grad = None




def _set_phase_modes(model: TDNJournal, phase: str) -> None:
    """Control dropout/batch behavior in modules that are frozen by a phase."""
    if phase == "warmup_forecast":
        model.driver_encoder.train()
        model.target_encoder.train()
        model.target_decoder.train()
        model.calendar_encoder.train()
        model.forecast_head.train()
        model.adversary.eval()
    elif phase == "adversary_fit":
        model.driver_encoder.eval()
        model.target_encoder.eval()
        model.target_decoder.eval()
        model.calendar_encoder.eval()
        model.forecast_head.eval()
        model.adversary.train()
    elif phase == "driver_max":
        model.driver_encoder.train()
        model.target_encoder.eval()
        model.target_decoder.eval()
        model.calendar_encoder.eval()
        model.forecast_head.eval()
        model.adversary.eval()
    elif phase == "forecast_adv_min":
        model.driver_encoder.eval()
        model.target_encoder.train()
        model.target_decoder.train()
        model.calendar_encoder.train()
        model.forecast_head.train()
        model.adversary.train()
    else:
        raise ValueError(f"Unknown optimization phase: {phase}")

def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach().square()))
    return float(np.sqrt(squared))


def driver_variance_floor_loss(
    representation: torch.Tensor, minimum_std: float = 0.10
) -> torch.Tensor:
    """Prevent a trivial constant driver representation during confusion."""
    batch_std = representation.std(dim=0, unbiased=False)
    return torch.relu(minimum_std - batch_std).square().mean()


@dataclass
class EvalResult:
    mse: float
    mae: float
    rmse: float
    inverse_mse: float
    inverse_mae: float
    inverse_rmse: float
    n_windows: int


@torch.no_grad()
def evaluate_tdn(
    model: TDNJournal,
    loader,
    dataset,
    device: torch.device,
    max_batches: int | None = None,
) -> EvalResult:
    model.eval()
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x_hist, y_hist, y_future, _, cal_future, _ = batch
        x_hist = x_hist.to(device=device, dtype=torch.float32)
        y_hist = y_hist.to(device=device, dtype=torch.float32)
        cal_future = cal_future.to(device=device, dtype=torch.float32)
        prediction, _ = model.forecast_autoregressive(
            x_history=x_hist,
            y_history=y_hist,
            calendar_future=cal_future,
            detach_driver=False,
        )
        predictions.append(prediction.cpu().numpy())
        truths.append(y_future.numpy())
    pred = np.concatenate(predictions, axis=0)
    truth = np.concatenate(truths, axis=0)
    error = pred - truth
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    pred_original = dataset.inverse_target(pred)
    truth_original = dataset.inverse_target(truth)
    inverse_error = pred_original - truth_original
    inverse_mse = float(np.mean(inverse_error**2))
    inverse_mae = float(np.mean(np.abs(inverse_error)))
    inverse_rmse = float(np.sqrt(inverse_mse))
    return EvalResult(
        mse=mse,
        mae=mae,
        rmse=rmse,
        inverse_mse=inverse_mse,
        inverse_mae=inverse_mae,
        inverse_rmse=inverse_rmse,
        n_windows=int(len(truth)),
    )


@dataclass
class LeakageResult:
    mse: float
    r2_mean: float
    r2_per_coordinate: list[float]
    corr2_mean: float
    corr2_per_coordinate: list[float]
    mse_per_coordinate: list[float]
    n_windows: int


@torch.no_grad()
def evaluate_history_summary_leakage(
    model: TDNJournal,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> LeakageResult:
    """Evaluate recoverability of S_Y(H_Y) from the current driver context.

    This is an online diagnostic using the adversary that participates in the
    minimax game. It is not interpreted as a proof of statistical independence.
    Lower R^2 / higher reconstruction MSE means less target-history summary is
    recoverable by the current adversary class.
    """
    model.eval()
    truths: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x_hist, y_hist, _, _, _, _ = batch
        x_hist = x_hist.to(device=device, dtype=torch.float32)
        y_hist = y_hist.to(device=device, dtype=torch.float32)
        summary = target_history_summary(y_hist)
        driver_context = model.encode_driver(x_hist)
        recovered = model.adversary_prediction(driver_context)
        truths.append(summary.detach().cpu())
        predictions.append(recovered.detach().cpu())
    truth = torch.cat(truths, dim=0).numpy()
    pred = torch.cat(predictions, dim=0).numpy()
    error = pred - truth
    mse_per = np.mean(error**2, axis=0)
    centered = truth - np.mean(truth, axis=0, keepdims=True)
    sse = np.sum(error**2, axis=0)
    sst = np.sum(centered**2, axis=0)
    r2 = np.where(sst > 1e-12, 1.0 - sse / sst, 0.0)
    corr2 = []
    for j in range(truth.shape[1]):
        if np.std(truth[:, j]) < 1e-12 or np.std(pred[:, j]) < 1e-12:
            corr2.append(0.0)
        else:
            c = float(np.corrcoef(truth[:, j], pred[:, j])[0, 1])
            corr2.append(c * c if np.isfinite(c) else 0.0)
    return LeakageResult(
        mse=float(np.mean(error**2)),
        r2_mean=float(np.mean(r2)),
        r2_per_coordinate=[float(x) for x in r2],
        corr2_mean=float(np.mean(corr2)),
        corr2_per_coordinate=[float(x) for x in corr2],
        mse_per_coordinate=[float(x) for x in mse_per],
        n_windows=int(len(truth)),
    )


@dataclass
class TrainResult:
    history: list[dict]
    best_validation_mse: float
    best_epoch: int
    minimum_validation_mse: float
    minimum_validation_epoch: int
    epochs_completed: int
    elapsed_seconds: float
    best_state: dict
    optimization_audit: dict
    checkpoint_selection: dict


def train_tdn(
    model: TDNJournal,
    train_loader,
    validation_loader,
    train_dataset,
    validation_dataset,
    device: torch.device,
    epochs: int = 20,
    warmup_epochs: int = 4,
    utility_learning_rate: float = 1e-3,
    driver_learning_rate: float = 5e-4,
    forecast_learning_rate: float = 1e-3,
    adversary_learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    adversarial_weight: float = 0.05,
    driver_utility_weight: float = 1.0,
    adversary_min_weight: float = 1.0,
    variance_weight: float = 0.01,
    minimum_driver_std: float = 0.10,
    driver_steps: int = 1,
    min_steps: int = 1,
    gradient_clip: float = 1.0,
    patience: int = 6,
    checkpoint_selection_tolerance: float = 0.02,
    probe_max_train_samples: int = 16000,
    probe_max_validation_samples: int = 8000,
    probe_ridge_alpha: float = 1.0,
    probe_seed: int = 20260810,
    checkpoint_path: str | Path | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> TrainResult:
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be >=0 and < epochs")
    if driver_steps < 1 or min_steps < 1:
        raise ValueError("driver_steps and min_steps must be positive")

    model.to(device)
    criterion = nn.MSELoss()
    driver_parameters = model.driver_parameters()
    forecast_parameters = model.forecast_parameters()
    adversary_parameters = model.adversary_parameters()
    utility_parameters = [*driver_parameters, *forecast_parameters]

    utility_optimizer = torch.optim.AdamW(
        utility_parameters, lr=utility_learning_rate, weight_decay=weight_decay
    )
    driver_optimizer = torch.optim.AdamW(
        driver_parameters, lr=driver_learning_rate, weight_decay=weight_decay
    )
    forecast_optimizer = torch.optim.AdamW(
        forecast_parameters, lr=forecast_learning_rate, weight_decay=weight_decay
    )
    adversary_optimizer = torch.optim.AdamW(
        adversary_parameters, lr=adversary_learning_rate, weight_decay=weight_decay
    )

    best_validation = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    candidate_states: list[dict] = []
    warmup_boundary_state: dict | None = None
    maximum_forbidden_driver_gradient_in_min = 0.0
    maximum_forbidden_forecast_gradient_in_max = 0.0
    maximum_forbidden_adversary_gradient_in_max = 0.0
    maximum_driver_utility_gradient_in_max = 0.0
    maximum_driver_confusion_gradient_in_max = 0.0
    warmup_boundary_validation = None
    warmup_boundary_leakage = None
    start = perf_counter()

    for epoch in range(epochs):
        model.train()
        phase = "utility_warmup" if epoch < warmup_epochs else "decoupled_minimax"

        # Warm-up establishes a useful coordinate system but is not itself the
        # final adversarial model.  Reset validation checkpoint selection at the
        # minimax boundary so every eligible final checkpoint has completed at
        # least one true driver-MAX / forecast+adversary-MIN update cycle.
        if epoch == warmup_epochs:
            if history:
                warmup_boundary_validation = history[-1].get("validation_mse_autoregressive")
                warmup_boundary_leakage = {
                    "mse": history[-1].get("validation_history_summary_mse"),
                    "r2_mean": history[-1].get("validation_history_summary_r2_mean"),
                    "r2_per_coordinate": history[-1].get("validation_history_summary_r2_per_coordinate"),
                    "corr2_mean": history[-1].get("validation_history_summary_corr2_mean"),
                    "corr2_per_coordinate": history[-1].get("validation_history_summary_corr2_per_coordinate"),
                }
            warmup_boundary_state = deepcopy(model.state_dict())
            best_validation = float("inf")
            best_state = deepcopy(model.state_dict())
            best_epoch = 0
            stale_epochs = 0

        future_losses: list[float] = []
        adversary_losses: list[float] = []
        confusion_losses: list[float] = []
        driver_utility_losses: list[float] = []
        variance_losses: list[float] = []
        utility_gradient_norms: list[float] = []
        confusion_gradient_norms: list[float] = []
        weighted_utility_gradient_norms: list[float] = []
        weighted_confusion_gradient_norms: list[float] = []
        utility_confusion_cosines: list[float] = []
        weighted_confusion_to_utility_ratios: list[float] = []

        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            x_hist, y_hist, y_future, _, cal_future, _ = batch
            x_hist = x_hist.to(device=device, dtype=torch.float32)
            y_hist = y_hist.to(device=device, dtype=torch.float32)
            y_future = y_future.to(device=device, dtype=torch.float32)
            cal_future = cal_future.to(device=device, dtype=torch.float32)
            summary = target_history_summary(y_hist)

            if phase == "utility_warmup":
                # Establish a forecast-useful driver representation before the
                # adversarial game.  This is the only phase where future-target
                # loss is allowed to update the driver encoder.
                _set_phase_modes(model, "warmup_forecast")
                _set_requires_grad(driver_parameters, True)
                _set_requires_grad(forecast_parameters, True)
                _set_requires_grad(adversary_parameters, False)
                _clear_gradients(adversary_parameters)
                utility_optimizer.zero_grad(set_to_none=True)
                prediction, _ = model.forecast_teacher_forced(
                    x_hist, y_hist, y_future, cal_future, detach_driver=False
                )
                future_loss = criterion(prediction, y_future)
                future_loss.backward()
                torch.nn.utils.clip_grad_norm_(utility_parameters, gradient_clip)
                utility_optimizer.step()
                future_losses.append(float(future_loss.detach()))

                # Fit the history decoder on the current detached driver state.
                _set_phase_modes(model, "adversary_fit")
                _set_requires_grad(driver_parameters, False)
                _set_requires_grad(forecast_parameters, False)
                _set_requires_grad(adversary_parameters, True)
                adversary_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    driver_context = model.encode_driver(x_hist)
                adversary_loss = criterion(
                    model.adversary_prediction(driver_context.detach()), summary
                )
                adversary_loss.backward()
                torch.nn.utils.clip_grad_norm_(adversary_parameters, gradient_clip)
                adversary_optimizer.step()
                adversary_losses.append(float(adversary_loss.detach()))
                continue

            # -------------------------------------------------------------
            # MAX: update ONLY the driver encoder to make S_Y(H_Y) harder to
            # recover.  Forecast and history-decoder parameters are frozen.
            # -------------------------------------------------------------
            for _ in range(driver_steps):
                _set_phase_modes(model, "driver_max")
                _set_requires_grad(driver_parameters, True)
                _set_requires_grad(forecast_parameters, False)
                _set_requires_grad(adversary_parameters, False)
                _clear_gradients(forecast_parameters)
                _clear_gradients(adversary_parameters)
                driver_optimizer.zero_grad(set_to_none=True)

                # V1.4 utility-aware MAX: the *same driver-only optimizer*
                # receives one composite objective. Frozen forecast parameters
                # provide a utility gradient path; frozen adversary parameters
                # provide the confusion gradient path. No other parameter group
                # can move in this step.
                prediction, driver_context = model.forecast_teacher_forced(
                    x_hist, y_hist, y_future, cal_future, detach_driver=False
                )
                driver_utility_loss = criterion(prediction, y_future)
                recovered = model.adversary_prediction(driver_context)
                reconstruction_loss = criterion(recovered, summary)
                variance_loss = driver_variance_floor_loss(
                    driver_context, minimum_std=minimum_driver_std
                )

                # V1.4 gradient-balance audit.  We measure the two intended
                # driver directions on the same batch before composing them:
                #   g_u = grad L_future
                #   g_c = grad (-L_history)
                # This exposes whether adversarial pressure is negligible or
                # directly conflicts with forecast utility.  Diagnostics are
                # recorded once per epoch to keep training cost low.
                if batch_index == 0:
                    utility_grads = torch.autograd.grad(
                        driver_utility_loss,
                        driver_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    confusion_grads = torch.autograd.grad(
                        -reconstruction_loss,
                        driver_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    util_sq = 0.0
                    conf_sq = 0.0
                    dot = 0.0
                    for gu, gc in zip(utility_grads, confusion_grads):
                        if gu is None or gc is None:
                            continue
                        gu_d = gu.detach()
                        gc_d = gc.detach()
                        util_sq += float(torch.sum(gu_d.square()))
                        conf_sq += float(torch.sum(gc_d.square()))
                        dot += float(torch.sum(gu_d * gc_d))
                    utility_grad_norm = float(np.sqrt(util_sq))
                    confusion_grad_norm = float(np.sqrt(conf_sq))
                    denom = max(utility_grad_norm * confusion_grad_norm, 1e-12)
                    cosine = float(dot / denom)
                    weighted_utility = float(driver_utility_weight * utility_grad_norm)
                    weighted_confusion = float(adversarial_weight * confusion_grad_norm)
                    ratio = float(weighted_confusion / max(weighted_utility, 1e-12))
                    utility_gradient_norms.append(utility_grad_norm)
                    confusion_gradient_norms.append(confusion_grad_norm)
                    weighted_utility_gradient_norms.append(weighted_utility)
                    weighted_confusion_gradient_norms.append(weighted_confusion)
                    utility_confusion_cosines.append(cosine)
                    weighted_confusion_to_utility_ratios.append(ratio)
                    maximum_driver_utility_gradient_in_max = max(
                        maximum_driver_utility_gradient_in_max, utility_grad_norm
                    )
                    maximum_driver_confusion_gradient_in_max = max(
                        maximum_driver_confusion_gradient_in_max, confusion_grad_norm
                    )

                driver_objective = (
                    driver_utility_weight * driver_utility_loss
                    - adversarial_weight * reconstruction_loss
                    + variance_weight * variance_loss
                )
                driver_objective.backward()
                maximum_forbidden_forecast_gradient_in_max = max(
                    maximum_forbidden_forecast_gradient_in_max,
                    _gradient_norm(forecast_parameters),
                )
                maximum_forbidden_adversary_gradient_in_max = max(
                    maximum_forbidden_adversary_gradient_in_max,
                    _gradient_norm(adversary_parameters),
                )
                torch.nn.utils.clip_grad_norm_(driver_parameters, gradient_clip)
                driver_optimizer.step()
                driver_utility_losses.append(float(driver_utility_loss.detach()))
                confusion_losses.append(float(reconstruction_loss.detach()))
                variance_losses.append(float(variance_loss.detach()))

            # -------------------------------------------------------------
            # MIN: detach/freeze the driver.  Future-target loss updates the
            # target path and forecast decoder; history reconstruction updates
            # the adversarial decoder.  The driver receives exactly no gradient.
            # -------------------------------------------------------------
            for _ in range(min_steps):
                _set_phase_modes(model, "forecast_adv_min")
                _set_requires_grad(driver_parameters, False)
                _set_requires_grad(forecast_parameters, True)
                _set_requires_grad(adversary_parameters, True)
                _clear_gradients(driver_parameters)
                forecast_optimizer.zero_grad(set_to_none=True)
                adversary_optimizer.zero_grad(set_to_none=True)

                prediction, driver_context = model.forecast_teacher_forced(
                    x_hist, y_hist, y_future, cal_future, detach_driver=True
                )
                future_loss = criterion(prediction, y_future)
                recovered = model.adversary_prediction(driver_context.detach())
                adversary_loss = criterion(recovered, summary)
                min_objective = future_loss + adversary_min_weight * adversary_loss
                min_objective.backward()
                maximum_forbidden_driver_gradient_in_min = max(
                    maximum_forbidden_driver_gradient_in_min,
                    _gradient_norm(driver_parameters),
                )
                torch.nn.utils.clip_grad_norm_(forecast_parameters, gradient_clip)
                torch.nn.utils.clip_grad_norm_(adversary_parameters, gradient_clip)
                forecast_optimizer.step()
                adversary_optimizer.step()
                future_losses.append(float(future_loss.detach()))
                adversary_losses.append(float(adversary_loss.detach()))

        # Restore ordinary requires_grad state before evaluation/checkpointing.
        _set_requires_grad(driver_parameters, True)
        _set_requires_grad(forecast_parameters, True)
        _set_requires_grad(adversary_parameters, True)
        validation = evaluate_tdn(
            model,
            validation_loader,
            validation_dataset,
            device,
            max_batches=max_validation_batches,
        )
        leakage = evaluate_history_summary_leakage(
            model,
            validation_loader,
            device,
            max_batches=max_validation_batches,
        )
        row = {
            "epoch": epoch + 1,
            "phase": phase,
            "future_mse_teacher_forced": float(np.mean(future_losses)) if future_losses else None,
            "adversary_mse": float(np.mean(adversary_losses)) if adversary_losses else None,
            "driver_utility_mse_teacher_forced": (
                float(np.mean(driver_utility_losses)) if driver_utility_losses else None
            ),
            "driver_utility_gradient_norm_first_batch": (
                float(np.mean(utility_gradient_norms)) if utility_gradient_norms else None
            ),
            "driver_confusion_gradient_norm_first_batch": (
                float(np.mean(confusion_gradient_norms)) if confusion_gradient_norms else None
            ),
            "weighted_driver_utility_gradient_norm_first_batch": (
                float(np.mean(weighted_utility_gradient_norms)) if weighted_utility_gradient_norms else None
            ),
            "weighted_driver_confusion_gradient_norm_first_batch": (
                float(np.mean(weighted_confusion_gradient_norms)) if weighted_confusion_gradient_norms else None
            ),
            "utility_vs_confusion_gradient_cosine_first_batch": (
                float(np.mean(utility_confusion_cosines)) if utility_confusion_cosines else None
            ),
            "weighted_confusion_to_utility_gradient_ratio_first_batch": (
                float(np.mean(weighted_confusion_to_utility_ratios)) if weighted_confusion_to_utility_ratios else None
            ),
            "driver_confusion_reconstruction_mse": (
                float(np.mean(confusion_losses)) if confusion_losses else None
            ),
            "driver_variance_floor_loss": (
                float(np.mean(variance_losses)) if variance_losses else None
            ),
            "validation_mse_autoregressive": validation.mse,
            "validation_mae_autoregressive": validation.mae,
            "validation_history_summary_mse": leakage.mse,
            "validation_history_summary_r2_mean": leakage.r2_mean,
            "validation_history_summary_r2_per_coordinate": leakage.r2_per_coordinate,
            "validation_history_summary_corr2_mean": leakage.corr2_mean,
            "validation_history_summary_corr2_per_coordinate": leakage.corr2_per_coordinate,
            "validation_history_summary_mse_per_coordinate": leakage.mse_per_coordinate,
        }
        history.append(row)
        if phase == "decoupled_minimax":
            candidate_states.append(
                {
                    "epoch": int(epoch + 1),
                    "validation_mse": float(validation.mse),
                    "state": deepcopy(model.state_dict()),
                }
            )
        print(
            f"Epoch {epoch + 1:03d} [{phase}] "
            f"train_future={row['future_mse_teacher_forced']:.6f} "
            f"driver_utility={row['driver_utility_mse_teacher_forced'] if row['driver_utility_mse_teacher_forced'] is not None else float('nan'):.6f} "
            f"adv={row['adversary_mse']:.6f} "
            f"leak_corr2={leakage.corr2_mean:.4f} "
            f"val_mse={validation.mse:.6f} val_mae={validation.mae:.6f}"
        )

        if validation.mse < best_validation - 1e-10:
            best_validation = validation.mse
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale_epochs = 0
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            stale_epochs += 1
        if epoch + 1 >= warmup_epochs + 2 and stale_epochs >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # ------------------------------------------------------------------
    # V1.4 validation-balanced checkpoint selection.
    # Stage 1: preserve forecasting quality by admitting only minimax
    # checkpoints within a relative tolerance of the pure validation-MSE
    # optimum. Stage 2: among those near-optimal checkpoints, select the
    # smallest *independent Ridge-probe* corr^2. The training adversary is
    # deliberately not used for checkpoint selection.
    # ------------------------------------------------------------------
    if not candidate_states:
        raise RuntimeError("No minimax candidate checkpoints were collected")
    minimum_validation = float(min(c["validation_mse"] for c in candidate_states))
    minimum_epoch = int(
        min(candidate_states, key=lambda c: c["validation_mse"])["epoch"]
    )
    threshold = minimum_validation * (1.0 + float(checkpoint_selection_tolerance))
    eligible_candidates = [c for c in candidate_states if c["validation_mse"] <= threshold]
    candidate_probe_records: list[dict] = []
    for candidate in eligible_candidates:
        model.load_state_dict(candidate["state"])
        ridge_audit = evaluate_independent_probes(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            device=device,
            max_train_samples=probe_max_train_samples,
            max_validation_samples=probe_max_validation_samples,
            ridge_alpha=probe_ridge_alpha,
            include_mlp=False,
            include_hsic=False,
            probe_seed=probe_seed,
        )
        corr2 = float(ridge_audit.ridge["corr2_mean"])
        if not np.isfinite(corr2):
            corr2 = float("inf")
        candidate_probe_records.append(
            {
                "epoch": int(candidate["epoch"]),
                "validation_mse": float(candidate["validation_mse"]),
                "ridge_corr2_mean": corr2,
                "ridge_r2_mean": float(ridge_audit.ridge["r2_mean"]),
                "ridge_mse": float(ridge_audit.ridge["mse"]),
                "state": candidate["state"],
            }
        )
    selected = min(
        candidate_probe_records,
        key=lambda c: (c["ridge_corr2_mean"], c["validation_mse"]),
    )
    best_state = selected["state"]
    best_epoch = int(selected["epoch"])
    best_validation = float(selected["validation_mse"])

    # Full independent audits are computed only for the warm-up boundary and
    # the selected final checkpoint.  MLP and HSIC are audit evidence, not
    # additional selection knobs.
    warmup_independent_probe = None
    if warmup_boundary_state is not None:
        model.load_state_dict(warmup_boundary_state)
        warmup_independent_probe = evaluate_independent_probes(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            device=device,
            max_train_samples=probe_max_train_samples,
            max_validation_samples=probe_max_validation_samples,
            ridge_alpha=probe_ridge_alpha,
            include_mlp=True,
            include_hsic=True,
            probe_seed=probe_seed,
        ).to_dict()

    model.load_state_dict(best_state)
    selected_independent_probe = evaluate_independent_probes(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        device=device,
        max_train_samples=probe_max_train_samples,
        max_validation_samples=probe_max_validation_samples,
        ridge_alpha=probe_ridge_alpha,
        include_mlp=True,
        include_hsic=True,
        probe_seed=probe_seed,
    ).to_dict()

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)

    elapsed = perf_counter() - start
    checkpoint_selection = {
        "rule": "forecast-near-optimal then minimum independent Ridge-probe corr2",
        "relative_validation_tolerance": float(checkpoint_selection_tolerance),
        "minimum_validation_mse": float(minimum_validation),
        "minimum_validation_epoch": int(minimum_epoch),
        "eligibility_threshold_mse": float(threshold),
        "eligible_count": int(len(candidate_probe_records)),
        "selected_epoch": int(best_epoch),
        "selected_validation_mse": float(best_validation),
        "selected_ridge_corr2_mean": float(selected["ridge_corr2_mean"]),
        "candidates": [
            {k: v for k, v in c.items() if k != "state"}
            for c in sorted(candidate_probe_records, key=lambda x: x["epoch"])
        ],
        "warmup_independent_probe": warmup_independent_probe,
        "selected_independent_probe": selected_independent_probe,
        "probe_policy": {
            "selection_probe": "Ridge only",
            "selection_probe_alpha": float(probe_ridge_alpha),
            "audit_probes": ["Ridge", "fixed-capacity MLP", "normalized RBF-HSIC"],
            "max_train_samples": int(probe_max_train_samples),
            "max_validation_samples": int(probe_max_validation_samples),
            "probe_seed": int(probe_seed),
            "test_split_used": False,
        },
    }

    audit = {
        "optimization_contract": {
            "version": "v1.4-validation-balanced-minimax",
            "warmup": "future loss may update driver + forecast; adversary fits detached driver",
            "max_step": "driver-only composite objective = utility_weight*future_loss - adv_weight*history_reconstruction + variance_floor; forecast/adversary frozen",
            "min_step": "future forecast + history-summary reconstruction minimized; driver detached",
            "driver_utility_weight": float(driver_utility_weight),
            "adversarial_weight": float(adversarial_weight),
        },
        "maximum_forbidden_driver_gradient_in_min": float(
            maximum_forbidden_driver_gradient_in_min
        ),
        "maximum_forbidden_forecast_gradient_in_max": float(
            maximum_forbidden_forecast_gradient_in_max
        ),
        "maximum_forbidden_adversary_gradient_in_max": float(
            maximum_forbidden_adversary_gradient_in_max
        ),
        "maximum_driver_utility_gradient_in_max": float(
            maximum_driver_utility_gradient_in_max
        ),
        "maximum_driver_confusion_gradient_in_max": float(
            maximum_driver_confusion_gradient_in_max
        ),
        "warmup_boundary_validation_mse": warmup_boundary_validation,
        "warmup_boundary_history_summary_leakage": warmup_boundary_leakage,
    }
    return TrainResult(
        history=history,
        best_validation_mse=float(best_validation),
        best_epoch=int(best_epoch),
        minimum_validation_mse=float(minimum_validation),
        minimum_validation_epoch=int(minimum_epoch),
        epochs_completed=len(history),
        elapsed_seconds=float(elapsed),
        best_state=best_state,
        optimization_audit=audit,
        checkpoint_selection=checkpoint_selection,
    )


def save_training_record(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
