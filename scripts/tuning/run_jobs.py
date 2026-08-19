#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ.get('TDN_TUNING_ROOT', 'results/tuning'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--jobs',type=Path,required=True)
    ap.add_argument('--stage',required=True)
    ap.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--selector',default='results/selectors/selector_gcm.json')
    a=ap.parse_args()
    jobs=json.loads(a.jobs.read_text())
    gpus=[x.strip() for x in a.gpus.split(',') if x.strip()]
    if not gpus: raise SystemExit('No GPUs supplied')
    stage_root=ROOT/a.stage
    runs=stage_root/'runs'; logs=stage_root/'logs'
    runs.mkdir(parents=True,exist_ok=True); logs.mkdir(parents=True,exist_ok=True)
    queue=list(jobs); active={}; failures=[]

    def expected(job):
        c=job['config']
        return runs / f"gcm_L{c['seq_len']}_H{c['pred_len']}_seed{c['seed']}_{job['job_id']}" / 'metrics.json'

    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue: continue
            j=queue.pop(0); out=expected(j)
            if out.exists() and out.stat().st_size>0:
                # Only skip a completed run if the saved file really contains test metrics.
                try:
                    d=json.loads(out.read_text())
                    if d.get('test') is not None:
                        print(f"[SKIP] {j['job_id']} (test metrics already present)")
                        continue
                except Exception:
                    pass
            c=j['config']
            cmd=[sys.executable,'-u','-m','tdn.run_tdn_base','--selector-manifest',a.selector,
                '--seq-len',str(c['seq_len']),'--pred-len',str(c['pred_len']),'--batch-size',str(c['batch_size']),
                '--driver-hidden',str(c['driver_hidden']),'--target-hidden',str(c['target_hidden']),
                '--target-embedding',str(c['target_embedding']),'--calendar-hidden',str(c['calendar_hidden']),
                '--num-layers',str(c['num_layers']),'--driver-heads',str(c['driver_heads']),'--target-heads',str(c['target_heads']),
                '--dropout',str(c['dropout']),'--epochs',str(c['epochs']),'--warmup-epochs',str(c['warmup_epochs']),
                '--utility-lr',str(c['utility_lr']),'--driver-lr',str(c['driver_lr']),'--forecast-lr',str(c['forecast_lr']),
                '--adversary-lr',str(c['adversary_lr']),'--weight-decay',str(c['weight_decay']),
                '--adv-weight',str(c['adv_weight']),'--driver-utility-weight',str(c['driver_utility_weight']),
                '--adv-min-weight',str(c['adv_min_weight']),'--variance-weight',str(c['variance_weight']),
                '--minimum-driver-std',str(c['minimum_driver_std']),'--driver-steps',str(c['driver_steps']),'--min-steps',str(c['min_steps']),
                '--gradient-clip',str(c['gradient_clip']),'--patience',str(c['patience']),
                '--checkpoint-selection-tolerance',str(c['checkpoint_selection_tolerance']),
                '--seed',str(c['seed']),'--run-tag',j['job_id'],'--device','cuda','--output-dir',str(runs)]
            # Deliberately DO NOT pass --skip-test or --validation-only-data.
            # Every one of the 266 hyperparameter runs evaluates the full test split.
            env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu
            lf=(logs/f"{j['job_id']}.log").open('w')
            p=subprocess.Popen(cmd,stdout=lf,stderr=subprocess.STDOUT,env=env)
            active[gpu]=(p,j,lf,time.time())
            print(f"[GPU {gpu}] START {j['job_id']}")
        time.sleep(1)
        for gpu,(p,j,lf,t0) in list(active.items()):
            rc=p.poll()
            if rc is None: continue
            lf.close(); active.pop(gpu)
            if rc!=0:
                failures.append((j['job_id'],rc)); print(f"[GPU {gpu}] FAIL {j['job_id']} rc={rc}")
            else:
                print(f"[GPU {gpu}] DONE {j['job_id']} ({time.time()-t0:.1f}s)")
    if failures:
        print('Failures:',failures,file=sys.stderr); raise SystemExit(1)

if __name__=='__main__': main()
