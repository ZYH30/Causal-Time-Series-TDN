"""Journal TDN: dual-path forecasting with decoupled adversarial optimization.

The architectural contract intentionally keeps TDN's original core idea while
removing the ambiguous future-driver behavior in the conference code:

* Driver path: only observed histories of GCM-selected variables are encoded.
* Target path: the observed target history is encoded separately.
* Known-future path: deterministic calendar covariates are supplied at each
  forecast step; no realized future operational/meteorological driver is used.
* Late fusion: a fixed historical driver context is fused with the evolving
  target state and the known-future calendar state at every horizon.
* Adversary: predicts a configurable summary S_Y(H_Y) from the driver context.

The training file controls which parameter group can move in each minimax step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """LSTM followed by causal self-attention and residual normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by attention heads")
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(self, sequence: torch.Tensor):
        encoded, (hidden, cell) = self.lstm(sequence)
        attended, _ = self.attention(
            encoded,
            encoded,
            encoded,
            attn_mask=self.causal_mask(encoded.shape[1], encoded.device),
            need_weights=False,
        )
        attended = self.norm(encoded + self.dropout(attended))
        context = attended[:, -1, :]
        return attended, context, hidden, cell


@dataclass
class TargetInstanceStats:
    mean: torch.Tensor
    scale: torch.Tensor


class TDNJournal(nn.Module):
    """Temporal Decouple Network used by the journal Weather experiments."""

    def __init__(
        self,
        driver_dim: int,
        calendar_dim: int = 4,
        driver_hidden: int = 32,
        target_hidden: int = 32,
        target_embedding: int = 8,
        calendar_hidden: int = 8,
        num_layers: int = 2,
        driver_heads: int = 4,
        target_heads: int = 4,
        dropout: float = 0.1,
        summary_dim: int = 4,
    ) -> None:
        super().__init__()
        self.driver_dim = int(driver_dim)
        self.calendar_dim = int(calendar_dim)
        self.driver_hidden = int(driver_hidden)
        self.target_hidden = int(target_hidden)
        self.target_embedding_size = int(target_embedding)
        self.calendar_hidden = int(calendar_hidden)
        self.summary_dim = int(summary_dim)

        self.driver_encoder = TemporalEncoder(
            input_size=driver_dim,
            hidden_size=driver_hidden,
            num_layers=num_layers,
            heads=driver_heads,
            dropout=dropout,
        )
        self.target_input_embedding = nn.Linear(1, target_embedding)
        self.target_encoder = TemporalEncoder(
            input_size=target_embedding,
            hidden_size=target_hidden,
            num_layers=num_layers,
            heads=target_heads,
            dropout=dropout,
        )

        # Decoder evolves only the endogenous target state.  Exogenous historical
        # drivers enter through a fixed context; only deterministic known-future
        # calendar values can change along the forecast horizon.
        self.target_decoder = nn.LSTM(
            input_size=target_embedding,
            hidden_size=target_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.calendar_encoder = nn.Sequential(
            nn.Linear(calendar_dim, calendar_hidden),
            nn.GELU(),
            nn.LayerNorm(calendar_hidden),
        )
        fusion_dim = target_hidden + driver_hidden + calendar_hidden
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, max(32, fusion_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, fusion_dim), 1),
        )

        self.adversary = nn.Sequential(
            nn.LayerNorm(driver_hidden),
            nn.Linear(driver_hidden, driver_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(driver_hidden, summary_dim),
        )

    @staticmethod
    def _target_instance_normalize(y_history: torch.Tensor):
        mean = y_history.mean(dim=1, keepdim=True).detach()
        scale = y_history.var(dim=1, unbiased=False, keepdim=True).add(1e-5).sqrt().detach()
        normalized = (y_history - mean) / scale
        return normalized, TargetInstanceStats(mean=mean, scale=scale)

    @staticmethod
    def _normalize_with_stats(values: torch.Tensor, stats: TargetInstanceStats):
        return (values - stats.mean) / stats.scale

    @staticmethod
    def _denormalize(values: torch.Tensor, stats: TargetInstanceStats):
        return values * stats.scale + stats.mean

    def encode_driver(self, x_history: torch.Tensor) -> torch.Tensor:
        _, context, _, _ = self.driver_encoder(x_history)
        return context

    def encode_target(self, y_history: torch.Tensor):
        y_normalized, stats = self._target_instance_normalize(y_history)
        target_embedded = self.target_input_embedding(y_normalized)
        _, context, hidden, cell = self.target_encoder(target_embedded)
        return context, hidden, cell, stats, y_normalized

    def adversary_prediction(self, driver_context: torch.Tensor) -> torch.Tensor:
        return self.adversary(driver_context)

    def _fusion_prediction(
        self,
        target_states: torch.Tensor,
        driver_context: torch.Tensor,
        calendar_future: torch.Tensor,
    ) -> torch.Tensor:
        horizon = target_states.shape[1]
        driver = driver_context[:, None, :].expand(-1, horizon, -1)
        calendar = self.calendar_encoder(calendar_future)
        return self.forecast_head(torch.cat([target_states, driver, calendar], dim=-1))

    def forecast_teacher_forced(
        self,
        x_history: torch.Tensor,
        y_history: torch.Tensor,
        y_future: torch.Tensor,
        calendar_future: torch.Tensor,
        detach_driver: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized training forecast using standard teacher forcing."""

        driver_context = self.encode_driver(x_history)
        if detach_driver:
            driver_context = driver_context.detach()
        _, hidden, cell, stats, y_history_norm = self.encode_target(y_history)
        future_norm = self._normalize_with_stats(y_future, stats)
        decoder_input = torch.cat(
            [y_history_norm[:, -1:, :], future_norm[:, :-1, :]], dim=1
        )
        target_states, _ = self.target_decoder(
            self.target_input_embedding(decoder_input), (hidden, cell)
        )
        prediction_norm = self._fusion_prediction(
            target_states, driver_context, calendar_future
        )
        return self._denormalize(prediction_norm, stats), driver_context

    def forecast_autoregressive(
        self,
        x_history: torch.Tensor,
        y_history: torch.Tensor,
        calendar_future: torch.Tensor,
        detach_driver: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Leakage-free rolling forecast used for validation, test, and deployment."""

        driver_context = self.encode_driver(x_history)
        if detach_driver:
            driver_context = driver_context.detach()
        _, hidden, cell, stats, y_history_norm = self.encode_target(y_history)
        previous = y_history_norm[:, -1:, :]
        predictions: list[torch.Tensor] = []
        for horizon in range(calendar_future.shape[1]):
            target_state, (hidden, cell) = self.target_decoder(
                self.target_input_embedding(previous), (hidden, cell)
            )
            step_prediction = self._fusion_prediction(
                target_state,
                driver_context,
                calendar_future[:, horizon : horizon + 1, :],
            )
            predictions.append(step_prediction)
            previous = step_prediction
        normalized = torch.cat(predictions, dim=1)
        return self._denormalize(normalized, stats), driver_context

    def driver_parameters(self) -> list[nn.Parameter]:
        return list(self.driver_encoder.parameters())

    def forecast_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        modules: Iterable[nn.Module] = (
            self.target_input_embedding,
            self.target_encoder,
            self.target_decoder,
            self.calendar_encoder,
            self.forecast_head,
        )
        for module in modules:
            parameters.extend(module.parameters())
        return parameters

    def adversary_parameters(self) -> list[nn.Parameter]:
        return list(self.adversary.parameters())

    def non_adversary_parameters(self) -> list[nn.Parameter]:
        return [*self.driver_parameters(), *self.forecast_parameters()]
