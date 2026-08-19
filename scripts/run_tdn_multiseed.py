#!/usr/bin/env python3
"""Run the frozen final TDN configuration across horizons and random seeds."""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260810, 20260811, 20260812]
DEFAULT_HORIZONS = [96, 192, 720]

FROZEN = {
    "seq_len": 336,
    "batch_size": 128,
    "driver_hidden": 64,
    "target_hidden": 64,
    "target_embedding": 16,
    "calendar_hidden": 32,
    "horizon_hidden": 16,
    "num_layers": 2,
    "driver_heads": 4,
    "target_heads": 8,
    "dropout": 0.05,
    "epochs": 80,
    "warmup_epochs": 2,
    "utility_lr": 1e-3,
    "driver_lr": 1e-4,
    "forecast_lr": 1e-3,
    "adversary_lr": 1e-3,
    "weight_decay": 0.0,
    "adv_weight": 0.03,
    "driver_utility_weight": 1.0,
    "adv_min_weight": 1.0,
    "variance_weight": 0.01,
    "minimum_driver_std": 0.10,
    "driver_steps": 1,
    "min_steps": 1,
    "gradient_clip": 0.5,
    "patience": 12,
    "checkpoint_selection_tolerance": 0.02,
    "probe_max_train_samples": 16000,
    "probe_max_validation_samples": 8000,
    "probe_ridge_alpha": 1.0,
    "probe_seed": 20260810,
    "decoder_mode": "ar",
    "rollout_block_size": 48,
    "rollout_final_mix": 0.50,
    "rollout_ramp_epochs": 4,
    "direct_aux_weight": 0.0,
    "ar_aux_weight": 0.0,
}


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def variant_from_manifest(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("selector_variant", path.stem.replace("selector_", "")))


def run_dir(output_dir: Path, selector_manifest: Path, horizon: int, seed: int, tag: str) -> Path:
    variant = variant_from_manifest(selector_manifest)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
    return output_dir / f"{variant}_L{FROZEN['seq_len']}_H{horizon}_seed{seed}_{safe}"


def build_cmd(selector: Path, horizon: int, seed: int, output_dir: Path, tag: str, repro_mode: str) -> list[str]:
    c = FROZEN
    return [
        sys.executable, "-u", "-m", "tdn.run_tdn",
        "--data", "dataset/weather.csv",
        "--selector-manifest", str(selector),
        "--seq-len", str(c["seq_len"]),
        "--pred-len", str(horizon),
        "--batch-size", str(c["batch_size"]),
        "--driver-hidden", str(c["driver_hidden"]),
        "--target-hidden", str(c["target_hidden"]),
        "--target-embedding", str(c["target_embedding"]),
        "--calendar-hidden", str(c["calendar_hidden"]),
        "--horizon-hidden", str(c["horizon_hidden"]),
        "--num-layers", str(c["num_layers"]),
        "--driver-heads", str(c["driver_heads"]),
        "--target-heads", str(c["target_heads"]),
        "--dropout", str(c["dropout"]),
        "--epochs", str(c["epochs"]),
        "--warmup-epochs", str(c["warmup_epochs"]),
        "--utility-lr", str(c["utility_lr"]),
        "--driver-lr", str(c["driver_lr"]),
        "--forecast-lr", str(c["forecast_lr"]),
        "--adversary-lr", str(c["adversary_lr"]),
        "--weight-decay", str(c["weight_decay"]),
        "--adv-weight", str(c["adv_weight"]),
        "--driver-utility-weight", str(c["driver_utility_weight"]),
        "--adv-min-weight", str(c["adv_min_weight"]),
        "--variance-weight", str(c["variance_weight"]),
        "--minimum-driver-std", str(c["minimum_driver_std"]),
        "--driver-steps", str(c["driver_steps"]),
        "--min-steps", str(c["min_steps"]),
        "--gradient-clip", str(c["gradient_clip"]),
        "--patience", str(c["patience"]),
        "--checkpoint-selection-tolerance", str(c["checkpoint_selection_tolerance"]),
        "--probe-max-train-samples", str(c["probe_max_train_samples"]),
        "--probe-max-validation-samples", str(c["probe_max_validation_samples"]),
        "--probe-ridge-alpha", str(c["probe_ridge_alpha"]),
        "--probe-seed", str(c["probe_seed"]),
        "--decoder-mode", str(c["decoder_mode"]),
        "--rollout-block-size", str(c["rollout_block_size"]),
        "--rollout-final-mix", str(c["rollout_final_mix"]),
        "--rollout-ramp-epochs", str(c["rollout_ramp_epochs"]),
        "--direct-aux-weight", str(c["direct_aux_weight"]),
        "--ar-aux-weight", str(c["ar_aux_weight"]),
        "--seed", str(seed),
        "--run-tag", tag,
        "--repro-mode", repro_mode,
        "--device", "cuda",
        "--output-dir", str(output_dir),
    ]


def run_one(gpu: str, selector: Path, horizon: int, seed: int, output_dir: Path, log_dir: Path, tag: str, force: bool, repro_mode: str) -> None:
    expected = run_dir(output_dir, selector, horizon, seed, tag) / "metrics.json"
    if expected.exists() and not force:
        print(f"[SKIP] H={horizon} seed={seed}: {expected}", flush=True)
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"H{horizon}_seed{seed}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONHASHSEED"] = str(seed)
    if repro_mode == "strict":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["NVIDIA_TF32_OVERRIDE"] = "0"
    cmd = build_cmd(selector, horizon, seed, output_dir, tag, repro_mode)
    print(f"[RUN][GPU {gpu}] H={horizon} seed={seed}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"TDN H={horizon} seed={seed} failed; see {log}")
    print(f"[DONE][GPU {gpu}] H={horizon} seed={seed}", flush=True)


def collect(selector: Path, horizons: list[int], seeds: list[int], output_dir: Path, tag: str, report_path: Path) -> None:
    rows = []
    for h in horizons:
        for seed in seeds:
            p = run_dir(output_dir, selector, h, seed, tag) / "metrics.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            test = d["test"]
            rows.append({
                "horizon": h,
                "seed": seed,
                "mse": float(test["inverse_mse"]),
                "mae": float(test["inverse_mae"]),
                "rmse": float(test["inverse_rmse"]),
                "best_epoch": int(d["best_epoch"]),
                "gradient_audit": d.get("optimization_audit", {}),
                "metrics_path": os.path.relpath(p.resolve(), ROOT.resolve()),
            })
    summary = []
    for h in horizons:
        subset = [r for r in rows if r["horizon"] == h]
        entry = {"horizon": h, "n": len(subset)}
        for metric in ("mse", "mae", "rmse"):
            vals = [r[metric] for r in subset]
            entry[f"{metric}_mean"] = statistics.mean(vals)
            entry[f"{metric}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary.append(entry)
    payload = {"metric_scale": "original_OT_scale", "frozen_config": FROZEN, "per_seed": rows, "summary": summary}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Final TDN Multi-Seed Results — Original Weather Scale",
        "",
        "- Selector: GCM-10 unless another selector manifest was explicitly supplied.",
        "- Final architecture: `ar_b48_m050`.",
        "- Seeds: `20260810, 20260811, 20260812`.",
        "- All MSE/MAE/RMSE values below are on the original `OT` scale.",
        "",
        "| Horizon | MSE mean ± SD | MAE mean ± SD | RMSE mean ± SD |",
        "|---:|---:|---:|---:|",
    ]
    for x in summary:
        lines.append(f"| {x['horizon']} | {x['mse_mean']:.6f} ± {x['mse_sd']:.6f} | {x['mae_mean']:.6f} ± {x['mae_sd']:.6f} | {x['rmse_mean']:.6f} ± {x['rmse_sd']:.6f} |")
    lines += ["", "## Per-seed results", "", "| Horizon | Seed | MSE | MAE | RMSE |", "|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['horizon']} | {r['seed']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['rmse']:.6f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT] {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", type=Path, default=Path("results/selectors/selector_gcm.json"))
    ap.add_argument("--horizons", default="96,192,720")
    ap.add_argument("--seeds", default="20260810,20260811,20260812")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--output-dir", type=Path, default=Path("results/tdn/main"))
    ap.add_argument("--log-dir", type=Path, default=Path("results/tdn/logs"))
    ap.add_argument("--tag", default="paper_final")
    ap.add_argument("--report", type=Path, default=Path("results/tdn/TDN_Final_MultiSeed_OriginalScale_Report.md"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repro-mode", choices=["legacy", "strict"], default="legacy")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    published_root = (ROOT / 'published').resolve()
    for label, path in [('output-dir', args.output_dir), ('log-dir', args.log_dir), ('report', args.report)]:
        resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        try:
            resolved.relative_to(published_root)
        except ValueError:
            continue
        raise SystemExit(f'Refusing to write {label} inside published/: {path}')

    horizons = parse_ints(args.horizons)
    seeds = parse_ints(args.seeds)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    tasks = [(h, seed) for h in horizons for seed in seeds]
    if args.dry_run:
        print(f"tasks={len(tasks)}; gpus={gpus}")
        for h, seed in tasks:
            print(" ".join(build_cmd(args.selector, h, seed, args.output_dir, args.tag, args.repro_mode)))
        return
    if not args.selector.exists():
        raise SystemExit(f"Selector manifest not found: {args.selector}. Run scripts/run_gcm.sh first.")
    q: queue.Queue[tuple[int, int]] = queue.Queue()
    for task in tasks:
        q.put(task)
    errors: list[str] = []

    def worker(gpu: str) -> None:
        while True:
            try:
                h, seed = q.get_nowait()
            except queue.Empty:
                return
            try:
                run_one(gpu, args.selector, h, seed, args.output_dir, args.log_dir, args.tag, args.force, args.repro_mode)
            except Exception as exc:
                errors.append(str(exc))
                print(f"[ERROR] {exc}", flush=True)
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise SystemExit("\n".join(errors))
    collect(args.selector, horizons, seeds, args.output_dir, args.tag, args.report)


if __name__ == "__main__":
    main()
