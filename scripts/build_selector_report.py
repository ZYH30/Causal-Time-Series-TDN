#!/usr/bin/env python3
"""Build the final cardinality-matched selector report on the original OT scale."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

SELECTORS = ["gcm", "correlation", "mutual_information", "random_00", "random_01"]
SEEDS = [20260810, 20260811, 20260812]


def mean_sd(xs):
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/selector_study"))
    ap.add_argument("--output", type=Path, default=Path("results/selector_study/TDN_Selector_Study_OriginalScale_Report.md"))
    args = ap.parse_args()
    summaries = {}
    per_seed = []
    for sel in SELECTORS:
        p = args.root / sel / "report.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d["per_seed"]
        for r in rows:
            per_seed.append({"selector": sel, **r})
        summaries[sel] = {m: mean_sd([float(r[m]) for r in rows]) for m in ("mse", "mae", "rmse")}
    random_rows = [r for r in per_seed if r["selector"].startswith("random_")]
    summaries["random_pooled"] = {m: mean_sd([float(r[m]) for r in random_rows]) for m in ("mse", "mae", "rmse")}

    lines = [
        "# H96 Cardinality-Matched Driver-Selection Study — Original Scale",
        "",
        "All selectors admit exactly 10 physical variables. The TDN architecture, optimizer, seeds, and `ar_b48_m050` training contract are identical across selectors.",
        "All error metrics are on the original `OT` scale. The public report intentionally compares aggregated mean and sample SD across seeds; no failure-count statistic is used.",
        "",
        "| Selector | Runs | MSE mean ± SD | MAE mean ± SD | RMSE mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "gcm": "GCM-10", "correlation": "Correlation-10", "mutual_information": "MI-10",
        "random_00": "Random-10 #0", "random_01": "Random-10 #1", "random_pooled": "Random pooled",
    }
    for sel in [*SELECTORS, "random_pooled"]:
        s = summaries[sel]; n = 6 if sel == "random_pooled" else 3
        lines.append(f"| {labels[sel]} | {n} | {s['mse'][0]:.6f} ± {s['mse'][1]:.6f} | {s['mae'][0]:.6f} ± {s['mae'][1]:.6f} | {s['rmse'][0]:.6f} ± {s['rmse'][1]:.6f} |")
    lines += ["", "## Per-seed audit", "", "| Selector | Seed | MSE | MAE | RMSE |", "|---|---:|---:|---:|---:|"]
    for r in per_seed:
        lines.append(f"| {labels[r['selector']]} | {r['seed']} | {float(r['mse']):.6f} | {float(r['mae']):.6f} | {float(r['rmse']):.6f} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT] {args.output}")

if __name__ == "__main__":
    main()
