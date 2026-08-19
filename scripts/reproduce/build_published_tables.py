#!/usr/bin/env python3
"""Regenerate paper-facing original-scale tables from frozen baselines and checkpoint inference."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PUB=ROOT/'published/original_scale'
SEEDS=[20260810,20260811,20260812]
HORIZONS=[96,192,720]

def mean_sd(xs): return statistics.mean(xs), statistics.stdev(xs) if len(xs)>1 else 0.0

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--main-inference',type=Path,required=True)
    ap.add_argument('--selector-inference',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    main_inf=json.loads(a.main_inference.read_text())
    sel_inf=json.loads(a.selector_inference.read_text())
    if not main_inf.get('full_evaluation') or not sel_inf.get('full_evaluation'):
        raise SystemExit('Full checkpoint inference is required to regenerate published tables.')

    # Baselines: canonical v1.4.1 per-seed metrics. TDN rows are replaced by checkpoint inference.
    rows=[]
    for r in read_csv(PUB/'Weather_OriginalScale_Main_Benchmark_PerSeed.csv'):
        if r['model']=='TDN': continue
        rows.append({'model':r['model'],'horizon':int(r['horizon']),'seed':int(r['seed']),
                     'mse':float(r['mse_raw']),'mae':float(r['mae_raw']),'rmse':float(r['rmse_raw']),
                     'source':'frozen_v1.4.1_metrics'})
    for r in main_inf['rows']:
        rows.append({'model':'TDN','horizon':int(r['horizon']),'seed':int(r['seed']),
                     'mse':float(r['inverse_mse']),'mae':float(r['inverse_mae']),'rmse':float(r['inverse_rmse']),
                     'source':'checkpoint_inference'})
    models=sorted({r['model'] for r in rows})
    summary=[]
    for h in HORIZONS:
        hr=[]
        for model in models:
            rr=[r for r in rows if r['model']==model and r['horizon']==h]
            if len(rr)!=3: raise RuntimeError(f'Expected 3 runs for {model} H={h}, got {len(rr)}')
            item={'model':model,'horizon':h}
            for m in ['mse','mae','rmse']:
                item[m+'_mean'],item[m+'_sd']=mean_sd([x[m] for x in rr])
            hr.append(item)
        for metric in ['mse','mae']:
            rank={x['model']:i+1 for i,x in enumerate(sorted(hr,key=lambda z:z[metric+'_mean']))}
            for x in hr: x[metric+'_rank']=rank[x['model']]
        summary.extend(hr)
    md=['# Weather Main Benchmark — Reproduced Original-Scale Results','',
        'TDN rows are recomputed by checkpoint-only inference. Baseline rows are regenerated from the frozen v1.4.1 per-seed benchmark metrics; no model is retrained.','']
    for h in HORIZONS:
        md += [f'## H={h}','', '| Model | MSE mean ± SD | MAE mean ± SD | RMSE mean ± SD | MSE rank | MAE rank |','|---|---:|---:|---:|---:|---:|']
        for x in sorted([z for z in summary if z['horizon']==h],key=lambda z:z['mse_mean']):
            md.append(f"| {x['model']} | {x['mse_mean']:.6f} ± {x['mse_sd']:.6f} | {x['mae_mean']:.6f} ± {x['mae_sd']:.6f} | {x['rmse_mean']:.6f} ± {x['rmse_sd']:.6f} | {x['mse_rank']} | {x['mae_rank']} |")
    (a.output_dir/'Weather_Main_Benchmark_Reproduced.md').write_text('\n'.join(md)+'\n')
    with (a.output_dir/'Weather_Main_Benchmark_Reproduced.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','horizon','mse_mean','mse_sd','mae_mean','mae_sd','rmse_mean','rmse_sd','mse_rank','mae_rank']); w.writeheader(); w.writerows(summary)

    # Selector table: all rows come from checkpoint inference. Paper-facing interpretation uses mean ± sample SD only.
    srows=[{'selector':r['label'],'seed':int(r['seed']),'mse':float(r['inverse_mse']),'mae':float(r['inverse_mae']),'rmse':float(r['inverse_rmse'])} for r in sel_inf['rows']]
    order=['gcm','correlation','mutual_information','random_00','random_01']
    ssum=[]
    for sel in order:
        rr=[r for r in srows if r['selector']==sel]
        item={'selector':sel,'n_runs':len(rr)}
        for m in ['mse','mae','rmse']: item[m+'_mean'],item[m+'_sd']=mean_sd([x[m] for x in rr])
        ssum.append(item)
    rr=[r for r in srows if r['selector'].startswith('random_')]
    item={'selector':'random_pooled','n_runs':len(rr)}
    for m in ['mse','mae','rmse']: item[m+'_mean'],item[m+'_sd']=mean_sd([x[m] for x in rr])
    ssum.append(item)
    labels={'gcm':'GCM-10','correlation':'Correlation-10','mutual_information':'MI-10','random_00':'Random-10 #0','random_01':'Random-10 #1','random_pooled':'Random pooled'}
    smd=['# H96 Cardinality-Matched Selector Study — Reproduced Original-Scale Results','',
         'All rows are recomputed from the frozen selector-study checkpoints. The public comparison reports three-seed mean and sample SD; no failure-count statistic is used.','',
         '| Selector | Runs | MSE mean ± SD | MAE mean ± SD | RMSE mean ± SD |','|---|---:|---:|---:|---:|']
    for x in ssum:
        smd.append(f"| {labels[x['selector']]} | {x['n_runs']} | {x['mse_mean']:.6f} ± {x['mse_sd']:.6f} | {x['mae_mean']:.6f} ± {x['mae_sd']:.6f} | {x['rmse_mean']:.6f} ± {x['rmse_sd']:.6f} |")
    (a.output_dir/'Weather_Selector_Study_Reproduced.md').write_text('\n'.join(smd)+'\n')
    with (a.output_dir/'Weather_Selector_Study_Reproduced.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['selector','n_runs','mse_mean','mse_sd','mae_mean','mae_sd','rmse_mean','rmse_sd']); w.writeheader(); w.writerows(ssum)
    print(f'[OUTPUT] {a.output_dir}')

if __name__=='__main__': main()
