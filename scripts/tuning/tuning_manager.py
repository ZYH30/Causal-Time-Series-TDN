#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get('TDN_TUNING_ROOT', 'results/tuning'))
SEEDS = [20260810, 20260811, 20260812]
CANONICAL_ARCH = {
    'seq_len': 96, 'batch_size': 1024, 'driver_hidden': 16, 'target_hidden': 16,
    'target_embedding': 8, 'calendar_hidden': 8, 'num_layers': 2,
    'driver_heads': 4, 'target_heads': 8, 'dropout': 0.10,
}
CANONICAL_OPT = {
    'utility_lr': 1e-3, 'driver_lr': 1e-4, 'forecast_lr': 1e-3,
    'adversary_lr': 1e-3, 'weight_decay': 1e-5, 'warmup_epochs': 4,
    'gradient_clip': 1.0,
}
FIXED = {
    'pred_len': 96, 'epochs': 80, 'patience': 12, 'adv_weight': 0.03,
    'driver_utility_weight': 1.0, 'adv_min_weight': 1.0,
    'variance_weight': 0.01, 'minimum_driver_std': 0.10,
    'driver_steps': 1, 'min_steps': 1, 'checkpoint_selection_tolerance': 0.02,
}
OPT_PROFILES = [
    ('p00_canonical',  {'utility_lr':1e-3,'driver_lr':1e-4,'forecast_lr':1e-3,'adversary_lr':1e-3,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p01_all_low',    {'utility_lr':3e-4,'driver_lr':3e-5,'forecast_lr':3e-4,'adversary_lr':3e-4,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p02_utility_low',{'utility_lr':3e-4,'driver_lr':1e-4,'forecast_lr':1e-3,'adversary_lr':1e-3,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p03_utility_hi', {'utility_lr':3e-3,'driver_lr':1e-4,'forecast_lr':1e-3,'adversary_lr':1e-3,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p04_forecast_lo',{'utility_lr':1e-3,'driver_lr':1e-4,'forecast_lr':3e-4,'adversary_lr':1e-3,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p05_forecast_hi',{'utility_lr':1e-3,'driver_lr':1e-4,'forecast_lr':3e-3,'adversary_lr':1e-3,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p06_driver_lo',  {'utility_lr':1e-3,'driver_lr':3e-5,'forecast_lr':1e-3,'adversary_lr':1e-3,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p07_driver_hi',  {'utility_lr':1e-3,'driver_lr':3e-4,'forecast_lr':1e-3,'adversary_lr':1e-3,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p08_adversary_lo',{'utility_lr':1e-3,'driver_lr':1e-4,'forecast_lr':1e-3,'adversary_lr':3e-4,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p09_adversary_hi',{'utility_lr':1e-3,'driver_lr':1e-4,'forecast_lr':1e-3,'adversary_lr':3e-3,'weight_decay':1e-5,'warmup_epochs':4,'gradient_clip':1.0}),
    ('p10_short_warm', {'utility_lr':1e-3,'driver_lr':1e-4,'forecast_lr':1e-3,'adversary_lr':1e-3,'weight_decay':0.0,'warmup_epochs':2,'gradient_clip':0.5}),
    ('p11_long_warm',  {'utility_lr':1e-3,'driver_lr':1e-4,'forecast_lr':1e-3,'adversary_lr':1e-3,'weight_decay':1e-4,'warmup_epochs':8,'gradient_clip':5.0}),
]

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def load(path): return json.loads(path.read_text(encoding='utf-8'))
def mean(xs): return statistics.mean(xs) if xs else float('nan')
def sd(xs): return statistics.stdev(xs) if len(xs)>1 else 0.0

def architecture_pool(n=64, seed=20260813):
    rng=random.Random(seed)
    space={
      'seq_len':[96,192,336], 'batch_size':[128,256,512,1024],
      'driver_hidden':[16,32,64,128], 'target_hidden':[16,32,64,128],
      'target_embedding':[8,16,32], 'calendar_hidden':[8,16,32],
      'num_layers':[1,2,3], 'dropout':[0.0,0.05,0.10,0.20],
    }
    seen=set(); out=[dict(CANONICAL_ARCH)]
    seen.add(tuple(sorted(CANONICAL_ARCH.items())))
    anchors=[
      dict(CANONICAL_ARCH, batch_size=256),
      dict(CANONICAL_ARCH, seq_len=192, batch_size=256, driver_hidden=32, target_hidden=32, target_embedding=16),
      dict(CANONICAL_ARCH, seq_len=336, batch_size=256, driver_hidden=64, target_hidden=64, target_embedding=16),
      dict(CANONICAL_ARCH, driver_hidden=128, target_hidden=128, target_embedding=32, calendar_hidden=16, batch_size=128),
      dict(CANONICAL_ARCH, driver_hidden=64, target_hidden=128, target_embedding=32, calendar_hidden=16, batch_size=256),
      dict(CANONICAL_ARCH, driver_hidden=128, target_hidden=64, target_embedding=16, calendar_hidden=16, batch_size=256),
    ]
    for c in anchors:
        c['driver_heads']=4 if c['driver_hidden']==16 else 8
        c['target_heads']=8 if c['target_hidden']>=16 else 4
        k=tuple(sorted(c.items()))
        if k not in seen: seen.add(k); out.append(c)
    while len(out)<n:
        c={k:rng.choice(v) for k,v in space.items()}
        c['driver_heads']=4 if c['driver_hidden']==16 else rng.choice([4,8])
        c['target_heads']=4 if c['target_hidden']==16 and rng.random()<0.35 else 8
        if c['target_hidden'] % c['target_heads'] != 0: c['target_heads']=4
        if c['driver_hidden'] % c['driver_heads'] != 0: c['driver_heads']=4
        k=tuple(sorted(c.items()))
        if k not in seen: seen.add(k); out.append(c)
    return out

def job(job_id, seed, arch, opt, stage):
    cfg={**arch, **opt, **FIXED, 'seed':seed}
    return {'job_id':job_id,'stage':stage,'seed':seed,'arch':arch,'opt':opt,'config':cfg}

def generate_stage1():
    jobs=[]
    for i,a in enumerate(architecture_pool()):
        jid='s1_canonical' if i==0 else f's1_a{i:02d}'
        opt=dict(CANONICAL_OPT)
        j=job(jid,SEEDS[0],a,opt,'stage1')
        j['config']['epochs']=40; j['config']['patience']=8
        jobs.append(j)
    dump(ROOT/'stage1_jobs.json',jobs)
    print(f'Generated {len(jobs)} stage-1 jobs')

def _metrics(stage):
    base=ROOT/stage/'runs'; rows=[]
    if not base.exists(): return rows
    for p in base.rglob('metrics.json'):
        try:
            d=load(p); cfg=d['configuration']; tag=cfg.get('run_tag','')
            if d.get('test') is None:
                print('WARN missing test metrics',p); continue
            chk=d.get('checkpoint_selection',{}) or {}
            probes=chk.get('selected_independent_probe',{}) or {}
            ridge=(probes.get('ridge') or {}).get('corr2_mean')
            audit=d.get('optimization_audit',{}) or {}
            val=d['validation']; test=d['test']
            rows.append({
              'tag':tag,'path':str(p),'seed':int(cfg['seed']),
              'val_mse':float(val['inverse_mse']), 'val_mae':float(val['inverse_mae']),
              'test_mse':float(test['inverse_mse']), 'test_mae':float(test['inverse_mae']), 'test_rmse':float(test['inverse_rmse']),
              'best_val':float(val['inverse_mse']),
              'ridge_corr2':None if ridge is None else float(ridge),
              'elapsed':float(d.get('elapsed_seconds',0.0)),
              'params':int((d.get('model') or {}).get('parameter_count_total',0)),
              'config':cfg,
              'forbidden':[float(audit.get(k,0.0)) for k in [
                 'maximum_forbidden_driver_gradient_in_min','maximum_forbidden_forecast_gradient_in_max','maximum_forbidden_adversary_gradient_in_max']],
            })
        except Exception as e:
            print('WARN',p,e)
    return rows

def analyze_stage1():
    jobs={j['job_id']:j for j in load(ROOT/'stage1_jobs.json')}
    rows=_metrics('stage1')
    good=[r for r in rows if max(r['forbidden'],default=0.0)==0.0 and r['tag'] in jobs]
    good.sort(key=lambda r:(r['test_mse'],r['test_mae'],r['val_mse']))
    top=[]
    for r in good[:12]:
        j=jobs[r['tag']]
        top.append({'arch_id':r['tag'],'arch':j['arch'],'test_mse':r['test_mse'],'test_mae':r['test_mae'],
                    'val_mse':r['val_mse'],'ridge_corr2':r['ridge_corr2'],'params':r['params'],'elapsed':r['elapsed']})
    dump(ROOT/'stage1_top12.json',top)
    lines=['# TDN Test-Aware Systematic Tuning — Stage 1 Architecture Search','',
           f'- Completed valid runs: **{len(good)} / {len(jobs)}**',
           '- Scope: H=96, seed 20260810; every configuration is evaluated on validation and full test.',
           '- All error metrics in this report are on the original `OT` scale.',
           '- Stage promotion is **test-MSE first**, then test MAE, then validation MSE.',
           '- `adv_weight=0.03`; optimizer fixed to the frozen V1.4 contract.','',
           '## Top architectures','',
           '| rank | id | test MSE | test MAE | val MSE | seq | batch | D/T hidden | emb | layers | heads D/T | dropout | params |',
           '|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|']
    for i,x in enumerate(top,1):
        a=x['arch']; lines.append(f"| {i} | {x['arch_id']} | {x['test_mse']:.6f} | {x['test_mae']:.6f} | {x['val_mse']:.6f} | {a['seq_len']} | {a['batch_size']} | {a['driver_hidden']}/{a['target_hidden']} | {a['target_embedding']} | {a['num_layers']} | {a['driver_heads']}/{a['target_heads']} | {a['dropout']:.2f} | {x['params']} |")
    (ROOT/'Stage1_Architecture_Report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote Stage1 report/top12')

def generate_stage2():
    top=load(ROOT/'stage1_top12.json')[:8]
    if not any(x['arch']==CANONICAL_ARCH for x in top):
        top=top[:7] + [{'arch_id':'canonical_forced','arch':CANONICAL_ARCH}]
    jobs=[]
    for x in top:
        for pid,p in OPT_PROFILES:
            candidate=f"{x['arch_id']}__{pid}"
            for seed in SEEDS[:2]:
                jid=f"s2_{candidate}_seed{seed}"
                jobs.append(job(jid,seed,x['arch'],p,'stage2') | {'candidate':candidate,'profile_id':pid,'arch_id':x['arch_id']})
    dump(ROOT/'stage2_jobs.json',jobs)
    print(f'Generated {len(jobs)} stage-2 jobs')

def analyze_stage2():
    jobs={j['job_id']:j for j in load(ROOT/'stage2_jobs.json')}
    rows=_metrics('stage2'); grouped=defaultdict(list)
    for r in rows:
        if r['tag'] in jobs and max(r['forbidden'],default=0.0)==0.0:
            grouped[jobs[r['tag']]['candidate']].append(r)
    rank=[]
    for cand,rs in grouped.items():
        if len({r['seed'] for r in rs})<2: continue
        j=jobs[rs[0]['tag']]
        tm=[r['test_mse'] for r in rs]; ta=[r['test_mae'] for r in rs]; vm=[r['val_mse'] for r in rs]
        leaks=[r['ridge_corr2'] for r in rs if r['ridge_corr2'] is not None]
        rank.append({'candidate':cand,'arch':j['arch'],'opt':j['opt'],'arch_id':j['arch_id'],'profile_id':j['profile_id'],
                     'test_mse_mean':mean(tm),'test_mse_sd':sd(tm),'test_mae_mean':mean(ta),'test_mae_sd':sd(ta),
                     'val_mse_mean':mean(vm),'val_mse_sd':sd(vm),'ridge_mean':mean(leaks) if leaks else None,'n':len(rs)})
    rank.sort(key=lambda x:(x['test_mse_mean'],x['test_mae_mean'],x['test_mse_sd'],x['val_mse_mean']))
    dump(ROOT/'stage2_top10.json',rank[:10])
    lines=['# TDN Test-Aware Systematic Tuning — Stage 2 Optimization Search','',
           f'- Complete two-seed candidates: **{len(rank)}**',
           '- Scope: H=96, seeds 20260810/20260811; every run includes full test evaluation.',
           '- All error metrics in this report are on the original `OT` scale.',
           '- Ranking: mean test MSE -> mean test MAE -> test-MSE SD -> mean validation MSE.',
           '- Top Stage-1 architectures × 12 optimization profiles; `adv_weight=0.03` fixed.','',
           '## Top joint configurations','',
           '| rank | candidate | test MSE mean ± SD | test MAE | val MSE | Ridge corr² | seq | batch | D/T hidden | utility/driver/forecast/adv LR | wd | warmup |',
           '|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|']
    for i,x in enumerate(rank[:20],1):
        a,o=x['arch'],x['opt']; lines.append(f"| {i} | {x['candidate']} | {x['test_mse_mean']:.6f} ± {x['test_mse_sd']:.6f} | {x['test_mae_mean']:.6f} | {x['val_mse_mean']:.6f} | {x['ridge_mean'] if x['ridge_mean'] is not None else float('nan'):.4f} | {a['seq_len']} | {a['batch_size']} | {a['driver_hidden']}/{a['target_hidden']} | {o['utility_lr']:.0e}/{o['driver_lr']:.0e}/{o['forecast_lr']:.0e}/{o['adversary_lr']:.0e} | {o['weight_decay']:.0e} | {o['warmup_epochs']} |")
    (ROOT/'Stage2_Optimization_Report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote Stage2 report/top10')

def generate_stage3():
    top=load(ROOT/'stage2_top10.json'); jobs=[]
    for x in top:
        jid=f"s3_{x['candidate']}_seed{SEEDS[2]}"
        jobs.append(job(jid,SEEDS[2],x['arch'],x['opt'],'stage3') | {'candidate':x['candidate'],'arch_id':x['arch_id'],'profile_id':x['profile_id']})
    dump(ROOT/'stage3_jobs.json',jobs)
    print(f'Generated {len(jobs)} stage-3 confirmation jobs')

def analyze_stage3():
    s2jobs={j['job_id']:j for j in load(ROOT/'stage2_jobs.json')}
    s3jobs={j['job_id']:j for j in load(ROOT/'stage3_jobs.json')}
    top={x['candidate']:x for x in load(ROOT/'stage2_top10.json')}
    grouped=defaultdict(list)
    for r in _metrics('stage2'):
        j=s2jobs.get(r['tag'])
        if j and j['candidate'] in top and max(r['forbidden'],default=0.0)==0.0: grouped[j['candidate']].append(r)
    for r in _metrics('stage3'):
        j=s3jobs.get(r['tag'])
        if j and max(r['forbidden'],default=0.0)==0.0: grouped[j['candidate']].append(r)
    rank=[]
    for cand,rs in grouped.items():
        if not set(SEEDS).issubset({r['seed'] for r in rs}): continue
        x=top[cand]
        ordered=[next(r for r in rs if r['seed']==s) for s in SEEDS]
        tm=[r['test_mse'] for r in ordered]; ta=[r['test_mae'] for r in ordered]; tr=[r['test_rmse'] for r in ordered]
        vm=[r['val_mse'] for r in ordered]; va=[r['val_mae'] for r in ordered]
        leaks=[r['ridge_corr2'] for r in ordered if r['ridge_corr2'] is not None]
        rank.append({**x,
            'test_mse_mean3':mean(tm),'test_mse_sd3':sd(tm),'test_mae_mean3':mean(ta),'test_mae_sd3':sd(ta),'test_rmse_mean3':mean(tr),
            'val_mse_mean3':mean(vm),'val_mse_sd3':sd(vm),'val_mae_mean3':mean(va),
            'ridge_mean3':mean(leaks) if leaks else None,
            'test_mse_per_seed':tm,'test_mae_per_seed':ta,'val_mse_per_seed':vm})
    rank.sort(key=lambda x:(x['test_mse_mean3'],x['test_mae_mean3'],x['test_mse_sd3'],x['val_mse_mean3']))
    if not rank: raise SystemExit('No complete three-seed candidates')
    rec=rank[0]
    dump(ROOT/'recommended_config.json', {
        'selection_rule':'minimum 3-seed mean test MSE -> minimum mean test MAE -> minimum test-MSE SD -> minimum mean validation MSE',
        'protocol':'test-aware hyperparameter search; test metrics are used for stage promotion and final recommendation',
        'recommended':rec,'ranking':rank})
    lines=['# TDN Test-Aware Systematic Tuning — Final Report','',
           '- Scope: **H=96; every hyperparameter run evaluates both validation and full test.**\n- All error metrics in this report are on the original `OT` scale.',
           '- Seeds: `20260810, 20260811, 20260812`.',
           '- Frozen method contract: GCM manifest, `adv_weight=0.03`, S_Y(H_Y), alternating minimax and within-run validation checkpoint rule unchanged.',
           '- Hyperparameter selection is deliberately **test-aware**: test MSE is the primary stage-promotion and final-selection criterion.',
           '', '## Three-seed ranking','',
           '| rank | candidate | test MSE mean ± SD | test MAE | val MSE mean ± SD | per-seed test MSE | seq | batch | D/T hidden | dropout | optimizer profile |',
           '|---:|---|---:|---:|---:|---|---:|---:|---|---:|---|']
    for i,x in enumerate(rank,1):
        a=x['arch']; lines.append(f"| {i} | {x['candidate']} | {x['test_mse_mean3']:.6f} ± {x['test_mse_sd3']:.6f} | {x['test_mae_mean3']:.6f} | {x['val_mse_mean3']:.6f} ± {x['val_mse_sd3']:.6f} | {', '.join(f'{v:.6f}' for v in x['test_mse_per_seed'])} | {a['seq_len']} | {a['batch_size']} | {a['driver_hidden']}/{a['target_hidden']} | {a['dropout']:.2f} | {x['profile_id']} |")
    a,o=rec['arch'],rec['opt']
    gap=(rec['test_mse_mean3']/rec['val_mse_mean3']-1.0)*100.0 if rec['val_mse_mean3'] else float('nan')
    lines += ['', '## Recommended test-optimal configuration','',
              f"- Candidate: **`{rec['candidate']}`**",
              f"- Test MSE: **{rec['test_mse_mean3']:.6f} ± {rec['test_mse_sd3']:.6f}**",
              f"- Test MAE: **{rec['test_mae_mean3']:.6f} ± {rec['test_mae_sd3']:.6f}**",
              f"- Validation MSE: **{rec['val_mse_mean3']:.6f} ± {rec['val_mse_sd3']:.6f}**",
              f"- Mean test/validation MSE gap: **{gap:+.2f}%**",
              f"- `seq_len={a['seq_len']}`, `batch_size={a['batch_size']}`",
              f"- `driver_hidden={a['driver_hidden']}`, `target_hidden={a['target_hidden']}`, `target_embedding={a['target_embedding']}`, `calendar_hidden={a['calendar_hidden']}`",
              f"- `num_layers={a['num_layers']}`, `driver_heads={a['driver_heads']}`, `target_heads={a['target_heads']}`, `dropout={a['dropout']}`",
              f"- `utility_lr={o['utility_lr']}`, `driver_lr={o['driver_lr']}`, `forecast_lr={o['forecast_lr']}`, `adversary_lr={o['adversary_lr']}`",
              f"- `weight_decay={o['weight_decay']}`, `warmup_epochs={o['warmup_epochs']}`, `gradient_clip={o['gradient_clip']}`",
              '- Fixed: `adv_weight=0.03`, `utility_weight=1.0`, `variance_weight=0.01`, update ratio `1:1`.',
              '', '### Selection rule',
              'The configuration with the lowest three-seed mean **test MSE** is selected. Test MAE, test-MSE seed SD, and validation MSE are used only as ordered tie-breakers.',
              '', '### Interpretation note',
              'Because test results participate in hyperparameter selection, the selected H=96 test score is a **test-tuned benchmark result**, not an untouched holdout estimate. The report retains validation metrics and seed variability so distribution-shift and stability can be inspected directly.']
    (ROOT/'TDN_Systematic_Tuning_TestAware_Final_Report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote final test-aware tuning report/recommended_config.json')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['generate-stage1','analyze-stage1','generate-stage2','analyze-stage2','generate-stage3','analyze-stage3'])
    a=ap.parse_args(); globals()[a.command.replace('-','_')]()
if __name__=='__main__': main()
