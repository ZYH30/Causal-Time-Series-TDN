#!/usr/bin/env python3
"""Paper-facing runner for the revised TDN Weather experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tdn.data import build_weather_loaders
from tdn.model import TDNJournal
from tdn.training import (
    evaluate_tdn,
    save_training_record,
    set_reproducible_seed,
    train_tdn,
)


def load_selected_features(path: Path) -> tuple[list[str], str, str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "selected_features" in payload:
        features = list(payload["selected_features"])
        label = str(payload.get("selector", path.stem))
    elif "selected_pairs" in payload:
        features = []
        for pair in payload["selected_pairs"]:
            if pair["feature"] not in features:
                features.append(pair["feature"])
        label = "gcm"
    else:
        raise ValueError(f"Manifest has no selected_features/selected_pairs: {path}")
    if not features:
        raise ValueError(f"No selected features in {path}")
    variant = str(payload.get("selector_variant", path.stem.replace("selector_", "")))
    return features, label, variant, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/weather.csv"))
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--driver-hidden", type=int, default=32)
    parser.add_argument("--target-hidden", type=int, default=32)
    parser.add_argument("--target-embedding", type=int, default=8)
    parser.add_argument("--calendar-hidden", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--driver-heads", type=int, default=4)
    parser.add_argument("--target-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--utility-lr", type=float, default=1e-3)
    parser.add_argument("--driver-lr", type=float, default=5e-4)
    parser.add_argument("--forecast-lr", type=float, default=1e-3)
    parser.add_argument("--adversary-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--adv-weight", type=float, default=0.05)
    parser.add_argument(
        "--driver-utility-weight",
        type=float,
        default=1.0,
        help="V1.4 MAX-step weight on future-target utility while only the driver encoder is trainable.",
    )
    parser.add_argument("--adv-min-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--minimum-driver-std", type=float, default=0.10)
    parser.add_argument("--driver-steps", type=int, default=1)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--checkpoint-selection-tolerance", type=float, default=0.02,
                        help="Relative validation-MSE tolerance for V1.4 near-optimal checkpoint eligibility.")
    parser.add_argument("--probe-max-train-samples", type=int, default=16000)
    parser.add_argument("--probe-max-validation-samples", type=int, default=8000)
    parser.add_argument("--probe-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--probe-seed", type=int, default=20260810)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional sanitized tag appended to the run directory; useful for validation-only sweeps.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tuning/runs")
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate the test split; use for validation-only tuning.")
    parser.add_argument(
        "--validation-only-data",
        action="store_true",
        help="Build only train/validation datasets. Requires --skip-test and prevents test windows from being instantiated during tuning.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    selected_features, selector_label, selector_variant, selector_payload = load_selected_features(
        args.selector_manifest
    )
    device = resolve_device(args.device)
    if args.validation_only_data and not args.skip_test:
        raise ValueError("--validation-only-data requires --skip-test")
    requested_splits = ("train", "val") if args.validation_only_data else ("train", "val", "test")
    datasets, loaders = build_weather_loaders(
        csv_path=args.data,
        selected_features=selected_features,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=requested_splits,
    )
    model = TDNJournal(
        driver_dim=len(selected_features),
        calendar_dim=4,
        driver_hidden=args.driver_hidden,
        target_hidden=args.target_hidden,
        target_embedding=args.target_embedding,
        calendar_hidden=args.calendar_hidden,
        num_layers=args.num_layers,
        driver_heads=args.driver_heads,
        target_heads=args.target_heads,
        dropout=args.dropout,
        summary_dim=4,
    )

    run_name = f"{selector_variant}_L{args.seq_len}_H{args.pred_len}_seed{args.seed}"
    if args.run_tag:
        safe_tag = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in args.run_tag)
        run_name = f"{run_name}_{safe_tag}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pt"

    print(f"Device: {device}")
    print(f"Selector: {selector_label}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    if "test" in datasets:
        print(
            f"Windows train/val/test: {len(datasets['train'])}/"
            f"{len(datasets['val'])}/{len(datasets['test'])}"
        )
    else:
        print(f"Windows train/val: {len(datasets['train'])}/{len(datasets['val'])}; test not instantiated")

    train_result = train_tdn(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["val"],
        train_dataset=datasets["train"],
        validation_dataset=datasets["val"],
        device=device,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        utility_learning_rate=args.utility_lr,
        driver_learning_rate=args.driver_lr,
        forecast_learning_rate=args.forecast_lr,
        adversary_learning_rate=args.adversary_lr,
        weight_decay=args.weight_decay,
        adversarial_weight=args.adv_weight,
        driver_utility_weight=args.driver_utility_weight,
        adversary_min_weight=args.adv_min_weight,
        variance_weight=args.variance_weight,
        minimum_driver_std=args.minimum_driver_std,
        driver_steps=args.driver_steps,
        min_steps=args.min_steps,
        gradient_clip=args.gradient_clip,
        patience=args.patience,
        checkpoint_selection_tolerance=args.checkpoint_selection_tolerance,
        probe_max_train_samples=args.probe_max_train_samples,
        probe_max_validation_samples=args.probe_max_validation_samples,
        probe_ridge_alpha=args.probe_ridge_alpha,
        probe_seed=args.probe_seed,
        checkpoint_path=checkpoint,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
    )
    validation = evaluate_tdn(
        model,
        loaders["val"],
        datasets["val"],
        device,
        max_batches=args.max_validation_batches,
    )
    test = None
    if not args.skip_test:
        test = evaluate_tdn(
            model,
            loaders["test"],
            datasets["test"],
            device,
            max_batches=args.max_test_batches,
        )

    payload = {
        "run_name": run_name,
        "device": str(device),
        "selector_manifest": str(args.selector_manifest),
        "selector": selector_label,
        "selector_variant": selector_variant,
        "selected_features": selected_features,
        "selector_payload": selector_payload,
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "model": {
            "name": "TDNJournal-v1.4-validation-balanced-minimax",
            "parameter_count_total": int(sum(p.numel() for p in model.parameters())),
            "parameter_count_driver": int(sum(p.numel() for p in model.driver_parameters())),
            "parameter_count_forecast": int(sum(p.numel() for p in model.forecast_parameters())),
            "parameter_count_adversary": int(sum(p.numel() for p in model.adversary_parameters())),
            "summary_S_Y": ["last", "mean", "std", "last_minus_first"],
            "future_driver_contract": (
                "historical selected drivers are fixed across the horizon; only "
                "known-future calendar covariates vary by forecast step"
            ),
        },
        "training_history": train_result.history,
        "optimization_audit": train_result.optimization_audit,
        "best_validation_mse": train_result.best_validation_mse,
        "best_epoch": train_result.best_epoch,
        "minimum_validation_mse": train_result.minimum_validation_mse,
        "minimum_validation_epoch": train_result.minimum_validation_epoch,
        "checkpoint_selection": train_result.checkpoint_selection,
        "epochs_completed": train_result.epochs_completed,
        "elapsed_seconds": train_result.elapsed_seconds,
        "validation": validation.__dict__,
        "test": (test.__dict__ if test is not None else None),
    }
    save_training_record(run_dir / "metrics.json", payload)
    print(json.dumps({"validation": validation.__dict__, "test": (test.__dict__ if test is not None else None)}, indent=2))
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
