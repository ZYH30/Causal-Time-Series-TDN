"""Training loop for the targeted TDN V1.5 long-horizon stability study."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn

from .model_stability import TDNJournalStability
from .probes import evaluate_independent_probes
from .training import (
    TrainResult, evaluate_tdn, evaluate_history_summary_leakage,
    target_history_summary, driver_variance_floor_loss,
    _set_requires_grad, _clear_gradients, _gradient_norm,
)


def _set_phase_modes(model: TDNJournalStability, phase: str) -> None:
    forecast_modules = [
        model.target_input_embedding, model.target_encoder, model.target_decoder,
        model.calendar_encoder, model.forecast_head, model.horizon_encoder, model.direct_head,
    ]
    if phase == "warmup_forecast":
        model.driver_encoder.train(); model.adversary.eval()
        for m in forecast_modules: m.train()
    elif phase == "adversary_fit":
        model.driver_encoder.eval(); model.adversary.train()
        for m in forecast_modules: m.eval()
    elif phase == "driver_max":
        model.driver_encoder.train(); model.adversary.eval()
        for m in forecast_modules: m.eval()
    elif phase == "forecast_adv_min":
        model.driver_encoder.eval(); model.adversary.train()
        for m in forecast_modules: m.train()
    else:
        raise ValueError(phase)


def _training_losses(model, criterion, x_hist, y_hist, y_future, cal_future,
                     detach_driver, rollout_block_size, rollout_mix,
                     direct_aux_weight, ar_aux_weight):
    pred, driver_context, branches = model.forecast_training(
        x_hist, y_hist, y_future, cal_future,
        detach_driver=detach_driver,
        rollout_block_size=rollout_block_size,
        rollout_mix=rollout_mix,
    )
    final_loss = criterion(pred, y_future)
    direct_loss = criterion(branches["direct"], y_future)
    ar_loss = criterion(branches["ar"], y_future)
    objective = final_loss + float(direct_aux_weight) * direct_loss + float(ar_aux_weight) * ar_loss
    return objective, final_loss, direct_loss, ar_loss, driver_context


def _rollout_mix(epoch: int, warmup_epochs: int, final_mix: float, ramp_epochs: int) -> float:
    if epoch < warmup_epochs or final_mix <= 0.0:
        return 0.0
    if ramp_epochs <= 1:
        return float(final_mix)
    progress = min(1.0, (epoch - warmup_epochs + 1) / float(ramp_epochs))
    return float(final_mix) * progress


def train_tdn_stability(
    model: TDNJournalStability,
    train_loader, validation_loader, train_dataset, validation_dataset,
    device: torch.device,
    epochs: int = 80, warmup_epochs: int = 2,
    utility_learning_rate: float = 1e-3, driver_learning_rate: float = 1e-4,
    forecast_learning_rate: float = 1e-3, adversary_learning_rate: float = 1e-3,
    weight_decay: float = 0.0, adversarial_weight: float = 0.03,
    driver_utility_weight: float = 1.0, adversary_min_weight: float = 1.0,
    variance_weight: float = 0.01, minimum_driver_std: float = 0.10,
    driver_steps: int = 1, min_steps: int = 1, gradient_clip: float = 0.5,
    patience: int = 12, checkpoint_selection_tolerance: float = 0.02,
    probe_max_train_samples: int = 16000, probe_max_validation_samples: int = 8000,
    probe_ridge_alpha: float = 1.0, probe_seed: int = 20260810,
    rollout_block_size: int = 0, rollout_final_mix: float = 0.0,
    rollout_ramp_epochs: int = 4, direct_aux_weight: float = 0.0,
    ar_aux_weight: float = 0.0,
    checkpoint_path: str | Path | None = None,
    max_train_batches: int | None = None, max_validation_batches: int | None = None,
) -> TrainResult:
    if warmup_epochs < 0 or warmup_epochs >= epochs: raise ValueError("invalid warmup")
    if not 0.0 <= rollout_final_mix <= 1.0: raise ValueError("rollout_final_mix must be [0,1]")
    model.to(device); criterion = nn.MSELoss()
    driver_parameters=model.driver_parameters(); forecast_parameters=model.forecast_parameters(); adversary_parameters=model.adversary_parameters()
    utility_parameters=[*driver_parameters,*forecast_parameters]
    utility_optimizer=torch.optim.AdamW(utility_parameters,lr=utility_learning_rate,weight_decay=weight_decay)
    driver_optimizer=torch.optim.AdamW(driver_parameters,lr=driver_learning_rate,weight_decay=weight_decay)
    forecast_optimizer=torch.optim.AdamW(forecast_parameters,lr=forecast_learning_rate,weight_decay=weight_decay)
    adversary_optimizer=torch.optim.AdamW(adversary_parameters,lr=adversary_learning_rate,weight_decay=weight_decay)

    best_validation=float('inf'); best_state=deepcopy(model.state_dict()); best_epoch=0; stale_epochs=0
    history=[]; candidate_states=[]; warmup_boundary_state=None; warmup_boundary_validation=None; warmup_boundary_leakage=None
    max_forbidden_driver=0.0; max_forbidden_forecast=0.0; max_forbidden_adv=0.0; max_util_grad=0.0; max_conf_grad=0.0
    start=perf_counter()

    for epoch in range(epochs):
        phase='utility_warmup' if epoch < warmup_epochs else 'decoupled_minimax'
        current_rollout=_rollout_mix(epoch,warmup_epochs,rollout_final_mix,rollout_ramp_epochs)
        if epoch==warmup_epochs:
            if history:
                warmup_boundary_validation=history[-1].get('validation_mse_autoregressive')
                warmup_boundary_leakage={'mse':history[-1].get('validation_history_summary_mse'),'corr2_mean':history[-1].get('validation_history_summary_corr2_mean')}
            warmup_boundary_state=deepcopy(model.state_dict()); best_validation=float('inf'); best_state=deepcopy(model.state_dict()); best_epoch=0; stale_epochs=0

        future_losses=[]; direct_losses=[]; ar_losses=[]; adv_losses=[]; conf_losses=[]; driver_util_losses=[]; variance_losses=[]
        util_norms=[]; conf_norms=[]; weighted_ratios=[]; cosines=[]
        for batch_index,batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index>=max_train_batches: break
            x_hist,y_hist,y_future,_,cal_future,_=batch
            x_hist=x_hist.to(device=device,dtype=torch.float32); y_hist=y_hist.to(device=device,dtype=torch.float32)
            y_future=y_future.to(device=device,dtype=torch.float32); cal_future=cal_future.to(device=device,dtype=torch.float32)
            summary=target_history_summary(y_hist)

            if phase=='utility_warmup':
                _set_phase_modes(model,'warmup_forecast'); _set_requires_grad(driver_parameters,True); _set_requires_grad(forecast_parameters,True); _set_requires_grad(adversary_parameters,False)
                _clear_gradients(adversary_parameters); utility_optimizer.zero_grad(set_to_none=True)
                obj,fl,dl,al,_ctx=_training_losses(model,criterion,x_hist,y_hist,y_future,cal_future,False,0,0.0,direct_aux_weight,ar_aux_weight)
                obj.backward(); torch.nn.utils.clip_grad_norm_(utility_parameters,gradient_clip); utility_optimizer.step()
                future_losses.append(float(fl.detach())); direct_losses.append(float(dl.detach())); ar_losses.append(float(al.detach()))
                _set_phase_modes(model,'adversary_fit'); _set_requires_grad(driver_parameters,False); _set_requires_grad(forecast_parameters,False); _set_requires_grad(adversary_parameters,True)
                adversary_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(): ctx=model.encode_driver(x_hist)
                adv_loss=criterion(model.adversary_prediction(ctx.detach()),summary); adv_loss.backward(); torch.nn.utils.clip_grad_norm_(adversary_parameters,gradient_clip); adversary_optimizer.step(); adv_losses.append(float(adv_loss.detach()))
                continue

            for _ in range(driver_steps):
                _set_phase_modes(model,'driver_max'); _set_requires_grad(driver_parameters,True); _set_requires_grad(forecast_parameters,False); _set_requires_grad(adversary_parameters,False)
                _clear_gradients(forecast_parameters); _clear_gradients(adversary_parameters); driver_optimizer.zero_grad(set_to_none=True)
                util_obj,fl,dl,al,ctx=_training_losses(model,criterion,x_hist,y_hist,y_future,cal_future,False,rollout_block_size,current_rollout,direct_aux_weight,ar_aux_weight)
                rec=criterion(model.adversary_prediction(ctx),summary); var=driver_variance_floor_loss(ctx,minimum_driver_std)
                if batch_index==0:
                    ug=torch.autograd.grad(util_obj,driver_parameters,retain_graph=True,allow_unused=True)
                    cg=torch.autograd.grad(-rec,driver_parameters,retain_graph=True,allow_unused=True)
                    us=cs=dot=0.0
                    for a,b in zip(ug,cg):
                        if a is None or b is None: continue
                        a=a.detach(); b=b.detach(); us+=float(torch.sum(a.square())); cs+=float(torch.sum(b.square())); dot+=float(torch.sum(a*b))
                    un=float(np.sqrt(us)); cn=float(np.sqrt(cs)); cosine=float(dot/max(un*cn,1e-12)); ratio=float(adversarial_weight*cn/max(driver_utility_weight*un,1e-12))
                    util_norms.append(un); conf_norms.append(cn); cosines.append(cosine); weighted_ratios.append(ratio); max_util_grad=max(max_util_grad,un); max_conf_grad=max(max_conf_grad,cn)
                driver_obj=driver_utility_weight*util_obj-adversarial_weight*rec+variance_weight*var
                driver_obj.backward(); max_forbidden_forecast=max(max_forbidden_forecast,_gradient_norm(forecast_parameters)); max_forbidden_adv=max(max_forbidden_adv,_gradient_norm(adversary_parameters))
                torch.nn.utils.clip_grad_norm_(driver_parameters,gradient_clip); driver_optimizer.step()
                driver_util_losses.append(float(fl.detach())); conf_losses.append(float(rec.detach())); variance_losses.append(float(var.detach()))

            for _ in range(min_steps):
                _set_phase_modes(model,'forecast_adv_min'); _set_requires_grad(driver_parameters,False); _set_requires_grad(forecast_parameters,True); _set_requires_grad(adversary_parameters,True)
                _clear_gradients(driver_parameters); forecast_optimizer.zero_grad(set_to_none=True); adversary_optimizer.zero_grad(set_to_none=True)
                forecast_obj,fl,dl,al,ctx=_training_losses(model,criterion,x_hist,y_hist,y_future,cal_future,True,rollout_block_size,current_rollout,direct_aux_weight,ar_aux_weight)
                adv_loss=criterion(model.adversary_prediction(ctx.detach()),summary); obj=forecast_obj+adversary_min_weight*adv_loss
                obj.backward(); max_forbidden_driver=max(max_forbidden_driver,_gradient_norm(driver_parameters)); torch.nn.utils.clip_grad_norm_(forecast_parameters,gradient_clip); torch.nn.utils.clip_grad_norm_(adversary_parameters,gradient_clip)
                forecast_optimizer.step(); adversary_optimizer.step(); future_losses.append(float(fl.detach())); direct_losses.append(float(dl.detach())); ar_losses.append(float(al.detach())); adv_losses.append(float(adv_loss.detach()))

        _set_requires_grad(driver_parameters,True); _set_requires_grad(forecast_parameters,True); _set_requires_grad(adversary_parameters,True)
        val=evaluate_tdn(model,validation_loader,validation_dataset,device,max_batches=max_validation_batches)
        leak=evaluate_history_summary_leakage(model,validation_loader,device,max_batches=max_validation_batches)
        row={'epoch':epoch+1,'phase':phase,'rollout_mix':current_rollout,'rollout_block_size':int(rollout_block_size),
             'future_mse_training':float(np.mean(future_losses)) if future_losses else None,
             'direct_branch_mse_training':float(np.mean(direct_losses)) if direct_losses else None,
             'ar_branch_mse_training':float(np.mean(ar_losses)) if ar_losses else None,
             'adversary_mse':float(np.mean(adv_losses)) if adv_losses else None,
             'driver_utility_mse_training':float(np.mean(driver_util_losses)) if driver_util_losses else None,
             'weighted_confusion_to_utility_gradient_ratio_first_batch':float(np.mean(weighted_ratios)) if weighted_ratios else None,
             'utility_vs_confusion_gradient_cosine_first_batch':float(np.mean(cosines)) if cosines else None,
             'validation_mse_autoregressive':val.mse,'validation_mae_autoregressive':val.mae,
             'validation_history_summary_mse':leak.mse,'validation_history_summary_r2_mean':leak.r2_mean,'validation_history_summary_r2_per_coordinate':leak.r2_per_coordinate,
             'validation_history_summary_corr2_mean':leak.corr2_mean,'validation_history_summary_corr2_per_coordinate':leak.corr2_per_coordinate,'validation_history_summary_mse_per_coordinate':leak.mse_per_coordinate}
        history.append(row)
        if phase=='decoupled_minimax': candidate_states.append({'epoch':epoch+1,'validation_mse':val.mse,'state':deepcopy(model.state_dict())})
        print(f"Epoch {epoch+1:03d} [{phase}] rollout={current_rollout:.2f}/B{rollout_block_size} train={row['future_mse_training']:.6f} val={val.mse:.6f}")
        if val.mse < best_validation-1e-10:
            best_validation=val.mse; best_state=deepcopy(model.state_dict()); best_epoch=epoch+1; stale_epochs=0
        else: stale_epochs+=1
        if epoch+1 >= warmup_epochs+2 and stale_epochs>=patience:
            print(f"Early stopping after {epoch+1} epochs"); break

    if not candidate_states: raise RuntimeError('No minimax candidates')
    min_val=min(c['validation_mse'] for c in candidate_states); min_epoch=min(candidate_states,key=lambda c:c['validation_mse'])['epoch']; threshold=min_val*(1+checkpoint_selection_tolerance)
    eligible=[c for c in candidate_states if c['validation_mse']<=threshold]; probe_records=[]
    for c in eligible:
        model.load_state_dict(c['state'])
        p=evaluate_independent_probes(model,train_dataset,validation_dataset,device,max_train_samples=probe_max_train_samples,max_validation_samples=probe_max_validation_samples,ridge_alpha=probe_ridge_alpha,include_mlp=False,include_hsic=False,probe_seed=probe_seed)
        corr=float(p.ridge['corr2_mean']); corr=corr if np.isfinite(corr) else float('inf')
        probe_records.append({'epoch':c['epoch'],'validation_mse':c['validation_mse'],'ridge_corr2_mean':corr,'ridge_r2_mean':float(p.ridge['r2_mean']),'ridge_mse':float(p.ridge['mse']),'state':c['state']})
    selected=min(probe_records,key=lambda c:(c['ridge_corr2_mean'],c['validation_mse'])); best_state=selected['state']; best_epoch=int(selected['epoch']); best_validation=float(selected['validation_mse'])
    warm_probe=None
    if warmup_boundary_state is not None:
        model.load_state_dict(warmup_boundary_state); warm_probe=evaluate_independent_probes(model,train_dataset,validation_dataset,device,max_train_samples=probe_max_train_samples,max_validation_samples=probe_max_validation_samples,ridge_alpha=probe_ridge_alpha,include_mlp=True,include_hsic=True,probe_seed=probe_seed).to_dict()
    model.load_state_dict(best_state); selected_probe=evaluate_independent_probes(model,train_dataset,validation_dataset,device,max_train_samples=probe_max_train_samples,max_validation_samples=probe_max_validation_samples,ridge_alpha=probe_ridge_alpha,include_mlp=True,include_hsic=True,probe_seed=probe_seed).to_dict()
    if checkpoint_path is not None:
        cp=Path(checkpoint_path); cp.parent.mkdir(parents=True,exist_ok=True); torch.save(best_state,cp)
    checkpoint_selection={'rule':'forecast-near-optimal then minimum independent Ridge-probe corr2','relative_validation_tolerance':checkpoint_selection_tolerance,'minimum_validation_mse':min_val,'minimum_validation_epoch':int(min_epoch),'eligibility_threshold_mse':threshold,'eligible_count':len(probe_records),'selected_epoch':best_epoch,'selected_validation_mse':best_validation,'selected_ridge_corr2_mean':selected['ridge_corr2_mean'],'candidates':[{k:v for k,v in c.items() if k!='state'} for c in probe_records],'warmup_independent_probe':warm_probe,'selected_independent_probe':selected_probe}
    audit={'optimization_contract':{'version':'v1.5-stability-study','decoder_mode':model.decoder_mode,'rollout_block_size':int(rollout_block_size),'rollout_final_mix':float(rollout_final_mix),'direct_aux_weight':float(direct_aux_weight),'ar_aux_weight':float(ar_aux_weight),'adversarial_weight':float(adversarial_weight)},
           'maximum_forbidden_driver_gradient_in_min':float(max_forbidden_driver),'maximum_forbidden_forecast_gradient_in_max':float(max_forbidden_forecast),'maximum_forbidden_adversary_gradient_in_max':float(max_forbidden_adv),'maximum_driver_utility_gradient_in_max':float(max_util_grad),'maximum_driver_confusion_gradient_in_max':float(max_conf_grad),'warmup_boundary_validation_mse':warmup_boundary_validation,'warmup_boundary_history_summary_leakage':warmup_boundary_leakage}
    return TrainResult(history=history,best_validation_mse=best_validation,best_epoch=best_epoch,minimum_validation_mse=float(min_val),minimum_validation_epoch=int(min_epoch),epochs_completed=len(history),elapsed_seconds=float(perf_counter()-start),best_state=best_state,optimization_audit=audit,checkpoint_selection=checkpoint_selection)
