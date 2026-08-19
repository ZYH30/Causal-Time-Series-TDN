#!/usr/bin/env python3
"""Build the paper-facing Weather benchmark report on the original OT scale."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

MODELS = [
    "TDN", "Autoformer", "Crossformer", "iTransformer", "MICN", "MultiPatchFormer",
    "Nonstationary_Transformer", "PatchTST", "Pyraformer", "SegRNN", "TimeMixer",
    "TimesNet", "TimeXer", "Transformer", "TSMixer",
]
HORIZONS = [96, 192, 720]
SEEDS = [20260810, 20260811, 20260812]


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdn-json", type=Path, default=Path("results/tdn/TDN_Final_MultiSeed_OriginalScale_Report.json"))
    ap.add_argument("--baseline-root", type=Path, default=Path("results/baselines/runs"))
    ap.add_argument("--output", type=Path, default=Path("Weather_Final_OriginalScale_Benchmark_Report.md"))
    args = ap.parse_args()

    tdn = json.loads(args.tdn_json.read_text(encoding="utf-8"))
    per = {("TDN", int(r["horizon"]), int(r["seed"])): r for r in tdn["per_seed"]}
    for model in MODELS[1:]:
        for h in HORIZONS:
            for seed in SEEDS:
                p = args.baseline_root / model / f"H{h}" / f"seed{seed}.json"
                d = json.loads(p.read_text(encoding="utf-8"))
                per[(model, h, seed)] = d

    summary = {}
    for model in MODELS:
        for h in HORIZONS:
            rows = [per[(model, h, seed)] for seed in SEEDS]
            summary[(model, h)] = {m: mean_sd([float(r[m]) for r in rows]) for m in ("mse", "mae", "rmse")}

    lines = [
        "# Weather Final Multi-Seed Benchmark — Original Scale",
        "",
        "- Seeds: `20260810, 20260811, 20260812`.",
        "- Horizons: `96, 192, 720`.",
        "- Baselines retain their frozen paper-facing model and training settings.",
        "- TDN uses GCM-10 and the frozen `ar_b48_m050` rollout-aware training contract.",
        "- All error metrics are reported on the original `OT` scale.",
    ]
    for h in HORIZONS:
        rows = []
        for model in MODELS:
            s = summary[(model, h)]
            rows.append((model, s))
        mse_rank = {model: i + 1 for i, (model, _) in enumerate(sorted(rows, key=lambda x: x[1]["mse"][0]))}
        mae_rank = {model: i + 1 for i, (model, _) in enumerate(sorted(rows, key=lambda x: x[1]["mae"][0]))}
        lines += [
            "", f"## H={h}", "",
            "| Model | MSE mean ± SD | MAE mean ± SD | RMSE mean ± SD | MSE rank | MAE rank |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model, s in sorted(rows, key=lambda x: x[1]["mse"][0]):
            mm, ms = s["mse"]; am, ass = s["mae"]; rm, rs = s["rmse"]
            lines.append(f"| {model} | {mm:.6f} ± {ms:.6f} | {am:.6f} ± {ass:.6f} | {rm:.6f} ± {rs:.6f} | {mse_rank[model]} | {mae_rank[model]} |")
    lines += ["", "## Final TDN per-seed results", "", "| H | Seed | MSE | MAE | RMSE |", "|---:|---:|---:|---:|---:|"]
    for h in HORIZONS:
        for seed in SEEDS:
            r = per[("TDN", h, seed)]
            lines.append(f"| {h} | {seed} | {float(r['mse']):.6f} | {float(r['mae']):.6f} | {float(r['rmse']):.6f} |")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT] {args.output}")

if __name__ == "__main__":
    main()
