"""Paper-facing Weather data pipeline for the journal TDN implementation.

The split reproduces ``Dataset_Custom`` from the supplied Time-Series-Library:
70% train / 10% validation / 20% test, with historical context allowed to reach
back across the split boundary while every forecast target remains inside its
own split.  All scalers are fit on the first 70% only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class WeatherSplit:
    train_rows: int
    validation_rows: int
    test_rows: int


def _calendar_features(stamps: pd.Series) -> np.ndarray:
    """Same four hour-level known-future features used by the original runner.

    The original TSlib runs use ``freq='h'`` even though Weather is sampled at
    ten-minute resolution.  We intentionally preserve that paper-facing input
    contract rather than adding a minute-of-hour feature only to TDN.
    """
    index = pd.DatetimeIndex(pd.to_datetime(stamps))
    return np.column_stack(
        [
            index.hour / 23.0 - 0.5,
            index.dayofweek / 6.0 - 0.5,
            (index.day - 1) / 30.0 - 0.5,
            (index.dayofyear - 1) / 365.0 - 0.5,
        ]
    ).astype(np.float32)


class WeatherTDNDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        selected_features: Sequence[str],
        seq_len: int,
        pred_len: int,
        split: str,
        target: str = "OT",
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.csv_path = Path(csv_path)
        self.selected_features = list(selected_features)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target = target
        self.split = split

        frame = pd.read_csv(self.csv_path)
        missing = [c for c in [*self.selected_features, target, "date"] if c not in frame]
        if missing:
            raise ValueError(f"Missing Weather columns: {missing}")

        n = len(frame)
        n_train = int(n * 0.70)
        n_test = int(n * 0.20)
        n_val = n - n_train - n_test
        self.split_sizes = WeatherSplit(n_train, n_val, n_test)
        borders = {
            "train": (0, n_train),
            "val": (n_train - self.seq_len, n_train + n_val),
            "test": (n - n_test - self.seq_len, n),
        }
        border1, border2 = borders[split]

        feature_scaler = StandardScaler().fit(
            frame.iloc[:n_train][self.selected_features].to_numpy(dtype=np.float64)
        )
        target_scaler = StandardScaler().fit(
            frame.iloc[:n_train][[target]].to_numpy(dtype=np.float64)
        )
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        feature_all = feature_scaler.transform(
            frame[self.selected_features].to_numpy(dtype=np.float64)
        ).astype(np.float32)
        target_all = target_scaler.transform(
            frame[[target]].to_numpy(dtype=np.float64)
        ).astype(np.float32)
        calendar_all = _calendar_features(frame["date"])

        self.features = feature_all[border1:border2]
        self.target_values = target_all[border1:border2]
        self.calendar = calendar_all[border1:border2]
        self.absolute_start = border1
        self.n_windows = len(self.features) - self.seq_len - self.pred_len + 1
        if self.n_windows <= 0:
            raise ValueError(
                f"No windows for split={split}, seq_len={seq_len}, pred_len={pred_len}"
            )

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int):
        history_start = int(index)
        history_end = history_start + self.seq_len
        future_end = history_end + self.pred_len
        x_hist = self.features[history_start:history_end]
        y_hist = self.target_values[history_start:history_end]
        y_future = self.target_values[history_end:future_end]
        cal_hist = self.calendar[history_start:history_end]
        cal_future = self.calendar[history_end:future_end]
        forecast_origin = self.absolute_start + history_end - 1
        return (
            torch.from_numpy(x_hist),
            torch.from_numpy(y_hist),
            torch.from_numpy(y_future),
            torch.from_numpy(cal_hist),
            torch.from_numpy(cal_future),
            torch.tensor(forecast_origin, dtype=torch.long),
        )

    def inverse_target(self, array: np.ndarray) -> np.ndarray:
        shape = array.shape
        flat = np.asarray(array).reshape(-1, 1)
        return self.target_scaler.inverse_transform(flat).reshape(shape)


def build_weather_loaders(
    csv_path: str | Path,
    selected_features: Sequence[str],
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int = 0,
    splits: Sequence[str] = ("train", "val", "test"),
    train_generator: torch.Generator | None = None,
) -> tuple[dict[str, WeatherTDNDataset], dict[str, DataLoader]]:
    requested = tuple(splits)
    invalid = [split for split in requested if split not in {"train", "val", "test"}]
    if invalid:
        raise ValueError(f"Unknown Weather split(s): {invalid}")
    datasets = {
        split: WeatherTDNDataset(
            csv_path=csv_path,
            selected_features=selected_features,
            seq_len=seq_len,
            pred_len=pred_len,
            split=split,
        )
        for split in requested
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=(train_generator if split == "train" else None),
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders
