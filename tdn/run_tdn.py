#!/usr/bin/env python3
"""Runner for the TDN V1.5 long-horizon stability architecture study."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from tdn.data import build_weather_loaders
from tdn.model_stability import TDNJournalStability
from tdn.training import evaluate_tdn, save_training_record, set_reproducible_seed
from tdn.training_stability import train_tdn_stability
from tdn.reproducibility import configure_reproducibility, make_dataloader_generator



def load_selected_features(path: Path) -> tuple[list[str], str, str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "selected_features" in payload:
        features = list(payload["selected_features"])
        label = str(payload.get("selector", path.stem))
    elif "selected_pairs" in payload:
        features = []
        for pair in payload["selected_pairs"]:
            if pair["feature"] not in features:
                features.append(pair["feature"])
        label = "gcm"
    else:
        raise ValueError(f"Manifest has no selected_features/selected_pairs: {path}")
    if not features:
        raise ValueError(f"No selected features in {path}")
    variant = str(payload.get("selector_variant", path.stem.replace("selector_", "")))
    return features, label, variant, payload


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('dataset/weather.csv')); p.add_argument('--selector-manifest',type=Path,required=True)
    p.add_argument('--seq-len',type=int,default=336); p.add_argument('--pred-len',type=int,required=True); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--driver-hidden',type=int,default=64); p.add_argument('--target-hidden',type=int,default=64); p.add_argument('--target-embedding',type=int,default=16); p.add_argument('--calendar-hidden',type=int,default=32); p.add_argument('--horizon-hidden',type=int,default=16)
    p.add_argument('--num-layers',type=int,default=2); p.add_argument('--driver-heads',type=int,default=4); p.add_argument('--target-heads',type=int,default=8); p.add_argument('--dropout',type=float,default=0.05)
    p.add_argument('--decoder-mode',choices=['ar','hybrid_constant','hybrid_linear'],default='ar'); p.add_argument('--direct-weight',type=float,default=0.25); p.add_argument('--direct-weight-start',type=float,default=0.10); p.add_argument('--direct-weight-end',type=float,default=0.50)
    p.add_argument('--rollout-block-size',type=int,default=0); p.add_argument('--rollout-final-mix',type=float,default=0.0); p.add_argument('--rollout-ramp-epochs',type=int,default=4); p.add_argument('--direct-aux-weight',type=float,default=0.0); p.add_argument('--ar-aux-weight',type=float,default=0.0)
    p.add_argument('--epochs',type=int,default=80); p.add_argument('--warmup-epochs',type=int,default=2); p.add_argument('--utility-lr',type=float,default=1e-3); p.add_argument('--driver-lr',type=float,default=1e-4); p.add_argument('--forecast-lr',type=float,default=1e-3); p.add_argument('--adversary-lr',type=float,default=1e-3); p.add_argument('--weight-decay',type=float,default=0.0)
    p.add_argument('--adv-weight',type=float,default=0.03); p.add_argument('--driver-utility-weight',type=float,default=1.0); p.add_argument('--adv-min-weight',type=float,default=1.0); p.add_argument('--variance-weight',type=float,default=0.01); p.add_argument('--minimum-driver-std',type=float,default=0.10); p.add_argument('--driver-steps',type=int,default=1); p.add_argument('--min-steps',type=int,default=1); p.add_argument('--gradient-clip',type=float,default=0.5); p.add_argument('--patience',type=int,default=12); p.add_argument('--checkpoint-selection-tolerance',type=float,default=0.02)
    p.add_argument('--probe-max-train-samples',type=int,default=16000); p.add_argument('--probe-max-validation-samples',type=int,default=8000); p.add_argument('--probe-ridge-alpha',type=float,default=1.0); p.add_argument('--probe-seed',type=int,default=20260810)
    p.add_argument('--seed',type=int,default=20260810); p.add_argument('--run-tag',type=str,required=True); p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda'); p.add_argument('--output-dir',type=Path,default=Path('results/tdn'))
    p.add_argument('--repro-mode',choices=['legacy','strict'],default='legacy')
    p.add_argument('--max-train-batches',type=int,default=None); p.add_argument('--max-validation-batches',type=int,default=None); p.add_argument('--max-test-batches',type=int,default=None)
    return p.parse_args()

def main():
    a=parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_abs = a.output_dir.resolve() if a.output_dir.is_absolute() else (repo_root / a.output_dir).resolve()
    published_abs = (repo_root / 'published').resolve()
    try:
        output_abs.relative_to(published_abs)
    except ValueError:
        pass
    else:
        raise RuntimeError('Refusing to write training outputs inside published/.')
    # legacy = exact v1.5.1 seed contract; strict = fully deterministic CUDA contract.
    repro = configure_reproducibility(a.seed, a.repro_mode)
    features,label,variant,payload=load_selected_features(a.selector_manifest); device=resolve_device(a.device)
    # Keep model initialization immediately after seeding. In strict mode, the
    # train sampler receives its own explicit RNG stream.
    model=TDNJournalStability(driver_dim=len(features),calendar_dim=4,driver_hidden=a.driver_hidden,target_hidden=a.target_hidden,target_embedding=a.target_embedding,calendar_hidden=a.calendar_hidden,num_layers=a.num_layers,driver_heads=a.driver_heads,target_heads=a.target_heads,dropout=a.dropout,summary_dim=4,decoder_mode=a.decoder_mode,direct_weight=a.direct_weight,direct_weight_start=a.direct_weight_start,direct_weight_end=a.direct_weight_end,horizon_hidden=a.horizon_hidden)
    train_generator = make_dataloader_generator(a.seed) if a.repro_mode == 'strict' else None
    datasets,loaders=build_weather_loaders(a.data,features,a.seq_len,a.pred_len,a.batch_size,a.num_workers,splits=('train','val','test'),train_generator=train_generator)
    safe=''.join(c if c.isalnum() or c in '-_.' else '_' for c in a.run_tag); run_name=f'{variant}_L{a.seq_len}_H{a.pred_len}_seed{a.seed}_{safe}'; run_dir=a.output_dir/run_name; run_dir.mkdir(parents=True,exist_ok=True)
    tr=train_tdn_stability(model,loaders['train'],loaders['val'],datasets['train'],datasets['val'],device,epochs=a.epochs,warmup_epochs=a.warmup_epochs,utility_learning_rate=a.utility_lr,driver_learning_rate=a.driver_lr,forecast_learning_rate=a.forecast_lr,adversary_learning_rate=a.adversary_lr,weight_decay=a.weight_decay,adversarial_weight=a.adv_weight,driver_utility_weight=a.driver_utility_weight,adversary_min_weight=a.adv_min_weight,variance_weight=a.variance_weight,minimum_driver_std=a.minimum_driver_std,driver_steps=a.driver_steps,min_steps=a.min_steps,gradient_clip=a.gradient_clip,patience=a.patience,checkpoint_selection_tolerance=a.checkpoint_selection_tolerance,probe_max_train_samples=a.probe_max_train_samples,probe_max_validation_samples=a.probe_max_validation_samples,probe_ridge_alpha=a.probe_ridge_alpha,probe_seed=a.probe_seed,rollout_block_size=a.rollout_block_size,rollout_final_mix=a.rollout_final_mix,rollout_ramp_epochs=a.rollout_ramp_epochs,direct_aux_weight=a.direct_aux_weight,ar_aux_weight=a.ar_aux_weight,checkpoint_path=run_dir/'best_model.pt',max_train_batches=a.max_train_batches,max_validation_batches=a.max_validation_batches)
    val=evaluate_tdn(model,loaders['val'],datasets['val'],device,max_batches=a.max_validation_batches); test=evaluate_tdn(model,loaders['test'],datasets['test'],device,max_batches=a.max_test_batches)
    cfg={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    rec={'run_name':run_name,'device':str(device),'selector':label,'selector_variant':variant,'selected_features':features,'selector_payload':payload,'configuration':cfg,'model':{'name':'TDNJournal-v1.5-stability-study','parameter_count_total':sum(p.numel() for p in model.parameters()),'decoder_mode':a.decoder_mode,'future_driver_contract':'historical selected drivers fixed; only known-future calendar varies'},'reproducibility':repro.to_dict(),'training_history':tr.history,'optimization_audit':tr.optimization_audit,'best_validation_mse':tr.best_validation_mse,'best_epoch':tr.best_epoch,'minimum_validation_mse':tr.minimum_validation_mse,'minimum_validation_epoch':tr.minimum_validation_epoch,'checkpoint_selection':tr.checkpoint_selection,'epochs_completed':tr.epochs_completed,'elapsed_seconds':tr.elapsed_seconds,'validation':val.__dict__,'test':test.__dict__}
    save_training_record(run_dir/'metrics.json',rec); print(json.dumps({'validation':val.__dict__,'test':test.__dict__},indent=2)); print('Saved',run_dir)
if __name__=='__main__': main()
