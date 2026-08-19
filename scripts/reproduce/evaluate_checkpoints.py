#!/usr/bin/env python3
"""Checkpoint-only evaluation for the published TDN and selector artifacts."""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tdn.data import build_weather_loaders
from tdn.model_stability import TDNJournalStability
from tdn.training import evaluate_tdn

PUBLISHED=ROOT/'published'
DATA=ROOT/'dataset/weather.csv'

def device_from_arg(value: str):
    if value=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    return torch.device(value)

def build_model(metrics: dict):
    c=metrics['configuration']
    return TDNJournalStability(
        driver_dim=len(metrics['selected_features']), calendar_dim=4,
        driver_hidden=int(c['driver_hidden']), target_hidden=int(c['target_hidden']),
        target_embedding=int(c['target_embedding']), calendar_hidden=int(c['calendar_hidden']),
        num_layers=int(c['num_layers']), driver_heads=int(c['driver_heads']),
        target_heads=int(c['target_heads']), dropout=float(c['dropout']), summary_dim=4,
        decoder_mode=str(c['decoder_mode']), direct_weight=float(c.get('direct_weight',0.25)),
        direct_weight_start=float(c.get('direct_weight_start',0.10)),
        direct_weight_end=float(c.get('direct_weight_end',0.50)),
        horizon_hidden=int(c.get('horizon_hidden',16)),
    )

def artifact_dirs(scope: str):
    if scope=='main':
        out=[]
        for h in [96,192,720]:
            for seed in [20260810,20260811,20260812]:
                out.append(('TDN',h,seed,PUBLISHED/f'tdn/H{h}/seed_{seed}'))
        return out
    sels=['gcm','correlation','mutual_information','random_00','random_01']
    out=[]
    for sel in sels:
        for seed in [20260810,20260811,20260812]:
            d = PUBLISHED/f'tdn/H96/seed_{seed}' if sel == 'gcm' else PUBLISHED/f'selector_study/checkpoints/{sel}/seed_{seed}'
            out.append((sel,96,seed,d))
    return out

def close(actual, expected, rtol, atol):
    return abs(actual-expected) <= atol + rtol*abs(expected)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scope',choices=['main','selector'],required=True)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--batch-size',type=int,default=None,help='Override evaluation batch size only.')
    ap.add_argument('--max-batches',type=int,default=None,help='Smoke-test option. Omit for full published evaluation.')
    ap.add_argument('--rtol',type=float,default=5e-5)
    ap.add_argument('--atol-raw',type=float,default=0.02)
    ap.add_argument('--allow-mismatch',action='store_true',help='Write audit output but do not fail when full inference exceeds the numerical tolerance.')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    dev=device_from_arg(a.device)
    rows=[]
    full=(a.max_batches is None)
    print(f'[INFO] device={dev}; scope={a.scope}; full_evaluation={full}')
    for label,h,seed,d in artifact_dirs(a.scope):
        metrics=json.loads((d/'metrics.json').read_text())
        cfg=metrics['configuration']; bs=a.batch_size or int(cfg['batch_size'])
        model=build_model(metrics)
        state=torch.load(d/'best_model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True); model.to(dev); model.eval()
        datasets,loaders=build_weather_loaders(DATA,metrics['selected_features'],int(cfg['seq_len']),int(cfg['pred_len']),bs,0,splits=('test',))
        got=evaluate_tdn(model,loaders['test'],datasets['test'],dev,max_batches=a.max_batches)
        exp=metrics['test']
        if full:
            checks={
                'mse':close(got.mse,float(exp['mse']),a.rtol,1e-7),
                'mae':close(got.mae,float(exp['mae']),a.rtol,1e-7),
                'rmse':close(got.rmse,float(exp['rmse']),a.rtol,1e-7),
                'inverse_mse':close(got.inverse_mse,float(exp['inverse_mse']),a.rtol,a.atol_raw),
                'inverse_mae':close(got.inverse_mae,float(exp['inverse_mae']),a.rtol,1e-3),
                'inverse_rmse':close(got.inverse_rmse,float(exp['inverse_rmse']),a.rtol,1e-3),
            }
        else:
            checks={'smoke_test_only':True}
        row={
            'scope':a.scope,'label':label,'horizon':h,'seed':seed,'device':str(dev),'full_evaluation':full,
            'mse':got.mse,'mae':got.mae,'rmse':got.rmse,
            'inverse_mse':got.inverse_mse,'inverse_mae':got.inverse_mae,'inverse_rmse':got.inverse_rmse,
            'n_windows':got.n_windows,
            'expected':{k:float(exp[k]) for k in ['mse','mae','rmse','inverse_mse','inverse_mae','inverse_rmse']},
            'checks':checks,
        }
        rows.append(row)
        status='PASS' if (not full or all(checks.values())) else 'MISMATCH'
        print(f"[{status}] {label} H={h} seed={seed}: raw MSE={got.inverse_mse:.6f}, MAE={got.inverse_mae:.6f}")
    bad=[]
    if full:
        bad=[r for r in rows if not all(r['checks'].values())]
        if bad:
            print('[ERROR] One or more checkpoint-inference metrics differ beyond the numerical tolerance.',file=sys.stderr)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    payload={'scope':a.scope,'device':str(dev),'full_evaluation':full,'all_checks_passed':not bad,'rows':rows}
    a.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'[OUTPUT] {a.output}')
    if bad and not a.allow_mismatch:
        raise SystemExit(2)

if __name__=='__main__': main()
