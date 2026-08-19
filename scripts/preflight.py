#!/usr/bin/env python3
"""Fast integrity check for the ICDE Weather reproduction package."""
from __future__ import annotations

import hashlib
import importlib
import platform
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_DATA_SHA256 = "f365909fa07b621e61a9a1293dc69ae803d3a865905066b481e7dcc6819932d7"
EXPECTED_CORE_HASHES = {
    "gcm/contract.py": "fd072f53c6fc902c204cd7ea210a04214c5c0473e12478f01030adef11821f11",
    "tdn/model_stability.py": "acb02a980309386e45219d7b9e9577db65b0fa53116bbd2f51564943514f3a29",
    "tdn/training_stability.py": "5e99d011bdb25efaae2154e08e010daa8546f004de1742bc48584716bef73e70",
    "baseline/run_seeded.py": "6f9be437b66eb3290c714b7e9e2b0c14595747eece9186fac87c9274c4d79c48",
}
BASELINES = [
    "Autoformer", "Crossformer", "iTransformer", "MICN", "MultiPatchFormer",
    "Nonstationary_Transformer", "PatchTST", "Pyraformer", "SegRNN", "TimeMixer",
    "TimesNet", "TimeXer", "Transformer", "TSMixer",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"[PASS] {message}")


def main() -> None:
    print(f"Python: {platform.python_version()}")
    data = ROOT / "dataset" / "weather.csv"
    check(data.exists(), "dataset/weather.csv exists")
    check(sha256(data) == EXPECTED_DATA_SHA256, "Weather dataset SHA256 matches the frozen benchmark")
    frame = pd.read_csv(data)
    check(frame.shape == (52696, 22), "Weather dataset shape is 52696 x 22")
    check("OT" in frame.columns and "date" in frame.columns, "Weather target and timestamp columns are present")
    n_train = int(len(frame) * 0.70)
    scaler = StandardScaler().fit(frame.iloc[:n_train][["OT"]].to_numpy())
    scale = float(scaler.scale_[0])
    check(abs(scale - 383.9570979658789) < 1e-9, "OT training-scale audit matches the frozen benchmark")
    print(f"OT training mean: {float(scaler.mean_[0]):.12f}")
    print(f"OT training scale: {scale:.12f}")

    for rel, expected in EXPECTED_CORE_HASHES.items():
        p = ROOT / rel
        check(p.exists(), f"{rel} exists")
        check(sha256(p) == expected, f"{rel} matches the frozen source hash")

    for model in BASELINES:
        check((ROOT / "baseline" / "models" / f"{model}.py").exists(), f"baseline model {model} is present")

    for module in ["gcm.contract", "gcm.run_gcm", "gcm.selectors", "tdn.model_stability", "tdn.training_stability"]:
        importlib.import_module(module)
        print(f"[PASS] import {module}")

    cjk = re.compile(r"[\u4e00-\u9fff]")
    offenders = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml"}:
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if cjk.search(text):
                offenders.append(str(p.relative_to(ROOT)))
    check(not offenders, f"English-only source/text audit ({len(offenders)} CJK-containing files)")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}; {props.total_memory / 1024**3:.2f} GiB")
        else:
            print("[INFO] CUDA is not available in this environment. GPU training was not executed by preflight.")
    except Exception as exc:
        print(f"[INFO] PyTorch CUDA inspection skipped: {exc}")

    print("PRE-FLIGHT PASSED")


if __name__ == "__main__":
    main()
