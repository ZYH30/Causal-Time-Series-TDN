from __future__ import annotations
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from tdn.model import TDNJournal
from tdn.model_stability import TDNJournalStability

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "published"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_weather_identity_and_shape():
    path = ROOT / "dataset/weather.csv"
    assert sha256(path) == "f365909fa07b621e61a9a1293dc69ae803d3a865905066b481e7dcc6819932d7"
    frame = pd.read_csv(path)
    assert frame.shape == (52696, 22)
    assert {"date", "OT"}.issubset(frame.columns)


def test_frozen_gcm_driver_pairs():
    manifest = json.loads((PUB / "gcm/weather_gcm_manifest.json").read_text())
    assert manifest["schema_version"] == "tdn-gcm-2.0"
    got = [(x["feature"], int(x["lag"])) for x in manifest["selected_pairs"]]
    expected = [
        ("Tlog_degC", 2), ("H2OC_m", 48), ("VPdef_mbar", 95),
        ("rho_g", 0), ("PAR_ol", 95), ("Tdew_degC", 0),
        ("p_mbar", 0), ("max_PAR", 95), ("raining_s", 95), ("wv_m", 95),
    ]
    assert got == expected


def test_selector_cardinality_is_matched():
    for name in ["gcm", "correlation", "mutual_information", "random_00", "random_01"]:
        payload = json.loads((PUB / f"selector_study/manifests/selector_{name}.json").read_text())
        assert len(payload["selected_features"]) == 10


def test_final_h96_contract_is_v151_ar_b48_m050():
    m = json.loads((PUB / "tdn/H96/seed_20260810/metrics.json").read_text())
    c = m["configuration"]
    assert c["pred_len"] == 96
    assert c["seq_len"] == 336
    assert c["batch_size"] == 128
    assert c["decoder_mode"] == "ar"
    assert c["rollout_block_size"] == 48
    assert c["rollout_final_mix"] == 0.5
    assert c["rollout_ramp_epochs"] == 4
    assert c["adv_weight"] == 0.03


def test_parameter_groups_are_disjoint():
    model = TDNJournal(driver_dim=10, driver_hidden=16, target_hidden=16, driver_heads=4, target_heads=4)
    groups = [model.driver_parameters(), model.forecast_parameters(), model.adversary_parameters()]
    ids = [set(map(id, group)) for group in groups]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])


def test_stability_model_shapes():
    model = TDNJournalStability(
        driver_dim=3, calendar_dim=4, driver_hidden=16, target_hidden=16,
        target_embedding=8, calendar_hidden=8, num_layers=1,
        driver_heads=4, target_heads=4, dropout=0.0,
        decoder_mode="ar", horizon_hidden=8,
    )
    x = torch.randn(2, 12, 3)
    yh = torch.randn(2, 12, 1)
    cal = torch.randn(2, 8, 4)
    pred, context = model.forecast_autoregressive(x, yh, cal)
    assert pred.shape == (2, 8, 1)
    assert context.shape == (2, 16)


def test_canonical_checkpoint_loads_strictly():
    metrics = json.loads((PUB / "tdn/H96/seed_20260810/metrics.json").read_text())
    c = metrics["configuration"]
    model = TDNJournalStability(
        driver_dim=len(metrics["selected_features"]), calendar_dim=4,
        driver_hidden=int(c["driver_hidden"]), target_hidden=int(c["target_hidden"]),
        target_embedding=int(c["target_embedding"]), calendar_hidden=int(c["calendar_hidden"]),
        num_layers=int(c["num_layers"]), driver_heads=int(c["driver_heads"]),
        target_heads=int(c["target_heads"]), dropout=float(c["dropout"]), summary_dim=4,
        decoder_mode=str(c["decoder_mode"]), direct_weight=float(c.get("direct_weight", 0.25)),
        direct_weight_start=float(c.get("direct_weight_start", 0.10)),
        direct_weight_end=float(c.get("direct_weight_end", 0.50)),
        horizon_hidden=int(c.get("horizon_hidden", 16)),
    )
    state = torch.load(PUB / "tdn/H96/seed_20260810/best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)


def test_v141_baseline_report_contains_no_tdn_row():
    text = (PUB / "baseline/Weather_Baseline_MultiSeed_Benchmark_Report.md").read_text()
    assert not any(line.startswith("| TDN |") for line in text.splitlines())
