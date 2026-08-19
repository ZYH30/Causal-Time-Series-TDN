"""Independent representation-leakage probes for TDN V1.4.

These probes are deliberately outside the adversarial minimax game.  They are
fit *after* a candidate model state has been frozen, using only train driver
representations and target-history summaries, and are evaluated on validation
representations.  They therefore measure recoverability without sharing the
adversary's optimization history.

Checkpoint selection uses the deterministic Ridge probe only.  The small MLP
probe and normalized RBF-HSIC are orthogonal audits reported for the warm-up
and selected checkpoints; they are not additional tuning objectives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset

from .model import TDNJournal


SUMMARY_NAMES = ["last", "mean", "std", "last_minus_first"]


def _target_history_summary(y_history: torch.Tensor) -> torch.Tensor:
    last = y_history[:, -1, 0]
    mean = y_history[:, :, 0].mean(dim=1)
    std = y_history[:, :, 0].std(dim=1, unbiased=False)
    trend = y_history[:, -1, 0] - y_history[:, 0, 0]
    return torch.stack([last, mean, std, trend], dim=1)


def _deterministic_subset(dataset, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    idx = np.linspace(0, len(dataset) - 1, num=max_samples, dtype=np.int64)
    # linspace can only repeat when max_samples > len(dataset), already handled.
    return Subset(dataset, idx.tolist())


@torch.no_grad()
def collect_driver_summary_arrays(
    model: TDNJournal,
    dataset,
    device: torch.device,
    max_samples: int | None = 16000,
    batch_size: int = 1024,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect frozen driver contexts Z_D and S_Y(H_Y) in chronological coverage."""
    model.eval()
    subset = _deterministic_subset(dataset, max_samples)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for batch in loader:
        x_hist, y_hist, *_ = batch
        x_hist = x_hist.to(device=device, dtype=torch.float32)
        y_hist = y_hist.to(device=device, dtype=torch.float32)
        z = model.encode_driver(x_hist)
        summary = _target_history_summary(y_hist)
        zs.append(z.detach().cpu().numpy())
        ys.append(summary.detach().cpu().numpy())
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def _regression_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    error = pred - truth
    mse_per = np.mean(error**2, axis=0)
    centered = truth - np.mean(truth, axis=0, keepdims=True)
    sse = np.sum(error**2, axis=0)
    sst = np.sum(centered**2, axis=0)
    r2 = np.where(sst > 1e-12, 1.0 - sse / sst, 0.0)
    corr2: list[float] = []
    for j in range(truth.shape[1]):
        if np.std(truth[:, j]) < 1e-12 or np.std(pred[:, j]) < 1e-12:
            corr2.append(0.0)
        else:
            c = float(np.corrcoef(truth[:, j], pred[:, j])[0, 1])
            corr2.append(c * c if np.isfinite(c) else 0.0)
    return {
        "mse": float(np.mean(error**2)),
        "mse_per_coordinate": [float(x) for x in mse_per],
        "r2_mean": float(np.mean(r2)),
        "r2_per_coordinate": [float(x) for x in r2],
        "corr2_mean": float(np.mean(corr2)),
        "corr2_per_coordinate": [float(x) for x in corr2],
    }


def fit_ridge_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Deterministic linear leakage probe used for checkpoint selection."""
    x_scaler = StandardScaler().fit(z_train)
    y_scaler = StandardScaler().fit(y_train)
    xtr = x_scaler.transform(z_train)
    xva = x_scaler.transform(z_val)
    ytr = y_scaler.transform(y_train)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(xtr, ytr)
    pred_std = model.predict(xva)
    pred = y_scaler.inverse_transform(pred_std)
    out = _regression_metrics(y_val, pred)
    out.update({"probe": "ridge", "alpha": float(alpha)})
    return out


class _MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    seed: int = 20260810,
    hidden_dim: int = 32,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> dict:
    """Fixed-capacity nonlinear probe; no validation tuning or early stopping."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x_scaler = StandardScaler().fit(z_train)
    y_scaler = StandardScaler().fit(y_train)
    xtr = x_scaler.transform(z_train).astype(np.float32)
    ytr = y_scaler.transform(y_train).astype(np.float32)
    xva = x_scaler.transform(z_val).astype(np.float32)

    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr)
    dataset = torch.utils.data.TensorDataset(xt, yt)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    probe = _MLPProbe(xtr.shape[1], hidden_dim, ytr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    probe.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    probe.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(xva), batch_size):
            xb = torch.from_numpy(xva[start : start + batch_size]).to(device)
            predictions.append(probe(xb).cpu().numpy())
    pred_std = np.concatenate(predictions, axis=0)
    pred = y_scaler.inverse_transform(pred_std)
    out = _regression_metrics(y_val, pred)
    out.update(
        {
            "probe": "mlp",
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
        }
    )
    return out


def _median_bandwidth(x: torch.Tensor) -> torch.Tensor:
    distances = torch.pdist(x, p=2)
    positive = distances[distances > 0]
    if len(positive) == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return torch.median(positive).clamp_min(1e-6)


def normalized_rbf_hsic(
    z: np.ndarray,
    y: np.ndarray,
    max_samples: int = 1024,
) -> float:
    """Normalized RBF-HSIC on a deterministic, evenly spaced validation subset."""
    n = min(len(z), len(y))
    if n < 4:
        return float("nan")
    if n > max_samples:
        idx = np.linspace(0, n - 1, num=max_samples, dtype=np.int64)
        z = z[idx]
        y = y[idx]
    zt = torch.as_tensor(z, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    zt = (zt - zt.mean(0, keepdim=True)) / zt.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    yt = (yt - yt.mean(0, keepdim=True)) / yt.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    sz = _median_bandwidth(zt)
    sy = _median_bandwidth(yt)
    dz = torch.cdist(zt, zt).square()
    dy = torch.cdist(yt, yt).square()
    kz = torch.exp(-dz / (2.0 * sz.square()))
    ky = torch.exp(-dy / (2.0 * sy.square()))
    # Double-center kernels in O(n^2) memory/time rather than explicitly
    # multiplying by an n x n centering matrix.
    kzc = kz - kz.mean(dim=0, keepdim=True) - kz.mean(dim=1, keepdim=True) + kz.mean()
    kyc = ky - ky.mean(dim=0, keepdim=True) - ky.mean(dim=1, keepdim=True) + ky.mean()
    numerator = torch.sum(kzc * kyc)
    denominator = torch.sqrt(torch.sum(kzc * kzc) * torch.sum(kyc * kyc)).clamp_min(1e-12)
    return float((numerator / denominator).item())


@dataclass
class IndependentProbeAudit:
    ridge: dict
    mlp: dict | None
    normalized_rbf_hsic: float | None
    train_samples: int
    validation_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_independent_probes(
    model: TDNJournal,
    train_dataset,
    validation_dataset,
    device: torch.device,
    max_train_samples: int = 16000,
    max_validation_samples: int = 8000,
    ridge_alpha: float = 1.0,
    include_mlp: bool = True,
    include_hsic: bool = True,
    probe_seed: int = 20260810,
) -> IndependentProbeAudit:
    ztr, ytr = collect_driver_summary_arrays(
        model, train_dataset, device, max_samples=max_train_samples
    )
    zva, yva = collect_driver_summary_arrays(
        model, validation_dataset, device, max_samples=max_validation_samples
    )
    ridge = fit_ridge_probe(ztr, ytr, zva, yva, alpha=ridge_alpha)
    mlp = (
        fit_mlp_probe(ztr, ytr, zva, yva, device=device, seed=probe_seed)
        if include_mlp
        else None
    )
    hsic = normalized_rbf_hsic(zva, yva) if include_hsic else None
    return IndependentProbeAudit(
        ridge=ridge,
        mlp=mlp,
        normalized_rbf_hsic=hsic,
        train_samples=int(len(ztr)),
        validation_samples=int(len(zva)),
    )
