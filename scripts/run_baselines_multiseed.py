#!/usr/bin/env python3
"""Run the exact paper-facing Weather baselines with the TDN seeds.

All baseline hyperparameters are transcribed from the supplied runAllExp.sh.
The only experimental changes are:
  1) run_seeded.py honors --seed (original run.py is untouched),
  2) model_id includes the seed so checkpoints/results do not collide,
  3) GPU assignment is externalized for parallel execution.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baseline"
OUT_ROOT = ROOT / "results" / "baselines" / "runs"
LOG_ROOT = ROOT / "results" / "baselines" / "logs"
RUN_TAG = "paper"

SEEDS_DEFAULT = [20260810, 20260811, 20260812]
HORIZONS_DEFAULT = [96, 192, 720]

COMMON = [
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--root_path", "../dataset/",
    "--data_path", "weather.csv",
    "--data", "custom",
    "--features", "MS",
    "--seq_len", "96",
]

# Exact settings from the supplied paper-facing runAllExp.sh.
SPECS: Dict[str, List[str]] = {
    "Autoformer": [
        "--label_len","48","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--des","Exp","--itr","1",
        "--train_epochs","2",
    ],
    "Crossformer": [
        "--label_len","48","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--d_model","32","--d_ff","32",
        "--top_k","5","--des","Exp","--itr","1",
    ],
    "iTransformer": [
        "--label_len","48","--e_layers","3","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--des","Exp",
        "--d_model","512","--d_ff","512","--itr","1",
    ],
    "MICN": [
        "--label_len","96","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--d_model","32","--d_ff","32",
        "--top_k","5","--des","Exp","--itr","1",
    ],
    "MultiPatchFormer": [
        "--label_len","48","--e_layers","1","--enc_in","21","--dec_in","21","--c_out","21",
        "--d_model","256","--d_ff","512","--des","Exp","--n_heads","8","--batch_size","32","--itr","1",
    ],
    "Nonstationary_Transformer": [
        "--label_len","48","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--des","Exp","--itr","1",
        "--train_epochs","3","--p_hidden_dims","256","256","--p_hidden_layers","2",
    ],
    "PatchTST": [
        "--label_len","48","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--des","Exp","--itr","1",
        "--n_heads","4","--train_epochs","3",
    ],
    "Pyraformer": [
        "--label_len","48","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--des","Exp","--itr","1","--train_epochs","2",
    ],
    "SegRNN": [
        "--pred_len_PLACEHOLDER", "",  # removed when building command; documents original local placement
        "--seg_len","48","--enc_in","21","--d_model","512","--dropout","0.5",
        "--learning_rate","0.0001","--des","Exp","--itr","1",
    ],
    "TimeMixer": [
        "--label_len","0","--e_layers","3","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--des","Exp","--itr","1",
        "--d_model","16","--d_ff","32","--batch_size","128","--learning_rate","0.01",
        "--train_epochs","20","--patience","10","--down_sampling_layers","3",
        "--down_sampling_method","avg","--down_sampling_window","2",
    ],
    "TimesNet": [
        "--label_len","48","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--d_model","32","--d_ff","32",
        "--top_k","5","--des","Exp","--itr","1",
    ],
    "TimeXer": [
        "--label_len","48","--e_layers","1","--factor","3","--enc_in","21","--dec_in","21","--c_out","21",
        "--des","Exp","--d_model","256","--d_ff","512","--batch_size","4","--itr","1",
    ],
    "Transformer": [
        "--label_len","48","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--des","Exp","--itr","1","--train_epochs","3",
    ],
    "TSMixer": [
        "--label_len","48","--e_layers","2","--d_layers","1","--factor","3",
        "--enc_in","21","--dec_in","21","--c_out","21","--d_model","32","--d_ff","32",
        "--top_k","5","--des","Exp","--itr","1",
    ],
}

MODEL_ORDER = list(SPECS.keys())


def weather_target_scale() -> float:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    frame = pd.read_csv(ROOT / "dataset" / "weather.csv")
    n_train = int(len(frame) * 0.70)
    scaler = StandardScaler().fit(frame.iloc[:n_train][["OT"]].to_numpy(dtype=np.float64))
    return float(scaler.scale_[0])


def clean_spec(spec: List[str]) -> List[str]:
    out=[]
    i=0
    while i < len(spec):
        if spec[i] == "--pred_len_PLACEHOLDER":
            i += 2
            continue
        out.append(spec[i]); i += 1
    return out


def build_cmd(model: str, horizon: int, seed: int, repro_mode: str = "legacy") -> List[str]:
    model_id = f"weather_96_{horizon}_seed{seed}_{RUN_TAG}"
    runner = "run_strict_seeded.py" if repro_mode == "strict" else "run_seeded.py"
    return [
        "python", "-u", runner,
        *COMMON,
        "--model_id", model_id,
        "--model", model,
        "--pred_len", str(horizon),
        *clean_spec(SPECS[model]),
        "--seed", str(seed),
        "--gpu", "0",
    ]


def result_json(model: str, horizon: int, seed: int) -> Path:
    return OUT_ROOT / model / f"H{horizon}" / f"seed{seed}.json"


def find_metrics(model: str, horizon: int, seed: int) -> Path:
    model_id = f"weather_96_{horizon}_seed{seed}_{RUN_TAG}"
    cands = list((BASELINE_ROOT / "results").glob(f"*{model_id}*{model}*/metrics.npy"))
    if not cands:
        raise FileNotFoundError(f"No metrics.npy found for {model} H={horizon} seed={seed}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def run_one(gpu: str, model: str, horizon: int, seed: int, force: bool, repro_mode: str) -> None:
    jout = result_json(model,horizon,seed)
    if jout.exists() and not force:
        print(f"[SKIP] {model} H={horizon} seed={seed}", flush=True)
        return
    jout.parent.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log = LOG_ROOT / f"{model}_H{horizon}_seed{seed}.log"
    cmd = build_cmd(model,horizon,seed,repro_mode)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONHASHSEED"] = str(seed)
    if repro_mode == "strict":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["NVIDIA_TF32_OVERRIDE"] = "0"
    print(f"[RUN][GPU {gpu}] {model} H={horizon} seed={seed}", flush=True)
    t0=time.time()
    with log.open("w", encoding="utf-8") as f:
        f.write("CMD: " + shlex.join(cmd) + "\n")
        f.flush()
        proc=subprocess.run(cmd, cwd=BASELINE_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    elapsed=time.time()-t0
    if proc.returncode != 0:
        raise RuntimeError(f"{model} H={horizon} seed={seed} failed; see {log}")
    mp=find_metrics(model,horizon,seed)
    a=np.load(mp)
    if a.shape[0] < 3:
        raise RuntimeError(f"Unexpected metrics in {mp}: {a}")
    scale = weather_target_scale()
    record={
        "model":model,"horizon":horizon,"seed":seed,
        "mse":float(a[1]) * scale * scale,
        "mae":float(a[0]) * scale,
        "rmse":float(a[2]) * scale,
        "metric_scale":"original_OT_scale",
        "standardized_audit":{"mae":float(a[0]),"mse":float(a[1]),"rmse":float(a[2])},
        "elapsed_seconds":elapsed,
        "seed_runner":"run_seeded.py",
        "baseline_settings_source":"runAllExp.sh",
        "only_protocol_change":"seed aligned to TDN; model_id made seed-unique for output isolation",
    }
    jout.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[DONE][GPU {gpu}] {model} H={horizon} seed={seed} raw_mse={record['mse']:.6f} raw_mae={record['mae']:.6f}", flush=True)


def worker(gpu: str, tasks: queue.Queue, force: bool, errors: list, repro_mode: str):
    while True:
        try: item=tasks.get_nowait()
        except queue.Empty: return
        try: run_one(gpu,*item,force,repro_mode)
        except Exception as e:
            errors.append((item,str(e)))
            print(f"[ERROR] {item}: {e}", flush=True)
        finally: tasks.task_done()


def parse_csv_int(s): return [int(x) for x in s.split(',') if x.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--seeds', default=','.join(map(str,SEEDS_DEFAULT)))
    ap.add_argument('--horizons', default=','.join(map(str,HORIZONS_DEFAULT)))
    ap.add_argument('--models', default=','.join(MODEL_ORDER))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--output-root', type=Path, default=None)
    ap.add_argument('--log-root', type=Path, default=None)
    ap.add_argument('--run-tag', default='paper')
    ap.add_argument('--repro-mode', choices=['legacy','strict'], default='legacy')
    ap.add_argument('--dry-run', action='store_true')
    args=ap.parse_args()
    global OUT_ROOT, LOG_ROOT, RUN_TAG
    if args.output_root is not None: OUT_ROOT = args.output_root.resolve()
    if args.log_root is not None: LOG_ROOT = args.log_root.resolve()
    RUN_TAG = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in args.run_tag)
    if 'published' in OUT_ROOT.parts or 'published' in LOG_ROOT.parts:
        raise SystemExit('Refusing to write baseline outputs inside published/.')
    gpus=[x.strip() for x in args.gpus.split(',') if x.strip()]
    seeds=parse_csv_int(args.seeds); horizons=parse_csv_int(args.horizons)
    models=[x.strip() for x in args.models.split(',') if x.strip()]
    unknown=[m for m in models if m not in SPECS]
    if unknown: raise SystemExit(f"Unknown models: {unknown}")

    tasks_list=[]
    # Interleave models/horizons/seeds to spread heavy and light models across GPUs.
    for h in horizons:
        for seed in seeds:
            for model in models:
                tasks_list.append((model,h,seed))
    if args.dry_run:
        print(f"Tasks: {len(tasks_list)}")
        for t in tasks_list[:20]: print(shlex.join(build_cmd(*t,args.repro_mode)))
        return

    OUT_ROOT.mkdir(parents=True,exist_ok=True); LOG_ROOT.mkdir(parents=True,exist_ok=True)
    q=queue.Queue()
    for t in tasks_list: q.put(t)
    errors=[]
    threads=[threading.Thread(target=worker,args=(gpu,q,args.force,errors,args.repro_mode),daemon=False) for gpu in gpus]
    for th in threads: th.start()
    for th in threads: th.join()
    if errors:
        print("\nFAILED TASKS:")
        for item,msg in errors: print(item,msg)
        raise SystemExit(1)
    print("[DONE] all requested baseline runs completed")

if __name__=='__main__': main()
