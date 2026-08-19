#!/usr/bin/env python3
"""Verify the immutable published artifact bundle and source-code freeze."""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "published"
EXPECTED_STAGING_SHA256 = "9ff1b899b169f711174ad4f89aa80388a2cc288fa3974068d06a7c444c3073aa"
EXPECTED_DATA_SHA256 = "f365909fa07b621e61a9a1293dc69ae803d3a865905066b481e7dcc6819932d7"
EXPECTED_CORE_HASHES = {
    "gcm/contract.py": "fd072f53c6fc902c204cd7ea210a04214c5c0473e12478f01030adef11821f11",
    "tdn/model_stability.py": "acb02a980309386e45219d7b9e9577db65b0fa53116bbd2f51564943514f3a29",
    "tdn/training_stability.py": "5e99d011bdb25efaae2154e08e010daa8546f004de1742bc48584716bef73e70",
    "baseline/run_seeded.py": "6f9be437b66eb3290c714b7e9e2b0c14595747eece9186fac87c9274c4d79c48",
}

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def check(ok: bool, msg: str):
    if not ok: raise RuntimeError(msg)
    print(f"[PASS] {msg}")

def main():
    check(PUBLISHED.exists(), "published artifact directory exists")
    check(sha256(ROOT/'dataset/weather.csv') == EXPECTED_DATA_SHA256, "root Weather dataset hash matches canonical data")
    prov=json.loads((PUBLISHED/'CANONICAL_PROVENANCE.json').read_text())
    check(prov['dataset_sha256']==EXPECTED_DATA_SHA256, "canonical provenance dataset hash matches")

    # Verify every public artifact listed by the release SHA256SUMS.
    for line in (PUBLISHED/'SHA256SUMS').read_text().splitlines():
        if not line.strip(): continue
        expected, rel = line.split(None,1)
        rel=rel.strip()
        p=PUBLISHED/rel
        check(p.exists(), f"published/{rel} exists")
        check(sha256(p)==expected, f"published/{rel} SHA256 matches")

    for run in prov['runs']:
        d=PUBLISHED/run['artifact_dir']
        check(sha256(d/'best_model.pt')==run['checkpoint_sha256'], f"{run['artifact_dir']} checkpoint hash matches")
        check(sha256(d/'metrics.json')==run['metrics_public_sha256'], f"{run['artifact_dir']} metrics hash matches")

    for rel, expected in EXPECTED_CORE_HASHES.items():
        check(sha256(ROOT/rel)==expected, f"{rel} matches frozen source hash")

    # Public repository hygiene.
    cjk=re.compile(r'[\u4e00-\u9fff]')
    private=re.compile(r'(/home/[^/\s]+/|/Users/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\)')
    bad_cjk=[]; bad_private=[]
    this_file = Path(__file__).resolve()
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts: continue
        if p.resolve() == this_file: continue  # avoid matching the audit regex literals in this script itself
        if p.suffix.lower() not in {'.py','.sh','.md','.txt','.json','.csv','.yaml','.yml','.toml','.ini','.cfg'}: continue
        text=p.read_text(encoding='utf-8',errors='replace')
        if cjk.search(text): bad_cjk.append(str(p.relative_to(ROOT)))
        if private.search(text): bad_private.append(str(p.relative_to(ROOT)))
    check(not bad_cjk, f"English-only text audit ({len(bad_cjk)} CJK-containing files)")
    check(not bad_private, f"no private user-home paths in public repository ({len(bad_private)} files)")

    baseline=(PUBLISHED/'baseline/Weather_Baseline_MultiSeed_Benchmark_Report.md').read_text()
    check(not any(x.startswith('| TDN |') for x in baseline.splitlines()), "v1.4.1 TDN rows are excluded from canonical baseline report")

    print("RELEASE VERIFICATION PASSED")
    print(f"Canonical staging archive SHA256 (handoff audit): {EXPECTED_STAGING_SHA256}")

if __name__=='__main__': main()
