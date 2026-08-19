"""TDN V1.5 stability-study model.

This module leaves the frozen V1.4 model untouched and adds two targeted
mechanisms for long-horizon stability:

1. Direct + autoregressive hybrid decoding.  A direct horizon head predicts
   each future step from target history context, driver history context,
   known-future calendar covariates, and an explicit horizon-position code.
   Its output is convexly blended with the recursive AR decoder.  The blended
   output is fed back to the AR decoder during inference, so the direct branch
   serves as a non-recursive anchor against rollout drift.
2. Block rollout-aware training.  Training can teacher-force within short
   blocks while allowing the model's own prediction to seed the next block.
   This exposes the decoder to self-generated boundary states without the cost
   of a fully step-wise training rollout over H=720.

No future unknown driver values are introduced.  Historical GCM-selected
variables remain fixed context; only known-future calendar covariates vary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import torch
from torch import nn

from .model import TemporalEncoder, TargetInstanceStats


class TDNJournalStability(nn.Module):
    def __init__(
        self,
        driver_dim: int,
        calendar_dim: int = 4,
        driver_hidden: int = 64,
        target_hidden: int = 64,
        target_embedding: int = 16,
        calendar_hidden: int = 32,
        num_layers: int = 2,
        driver_heads: int = 4,
        target_heads: int = 8,
        dropout: float = 0.05,
        summary_dim: int = 4,
        decoder_mode: str = "ar",
        direct_weight: float = 0.25,
        direct_weight_start: float = 0.10,
        direct_weight_end: float = 0.50,
        horizon_hidden: int = 16,
    ) -> None:
        super().__init__()
        if decoder_mode not in {"ar", "hybrid_constant", "hybrid_linear"}:
            raise ValueError(f"Unsupported decoder_mode={decoder_mode}")
        for name, value in {
            "direct_weight": direct_weight,
            "direct_weight_start": direct_weight_start,
            "direct_weight_end": direct_weight_end,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        self.driver_dim = int(driver_dim)
        self.calendar_dim = int(calendar_dim)
        self.driver_hidden = int(driver_hidden)
        self.target_hidden = int(target_hidden)
        self.target_embedding_size = int(target_embedding)
        self.calendar_hidden = int(calendar_hidden)
        self.summary_dim = int(summary_dim)
        self.decoder_mode = str(decoder_mode)
        self.direct_weight = float(direct_weight)
        self.direct_weight_start = float(direct_weight_start)
        self.direct_weight_end = float(direct_weight_end)
        self.horizon_hidden = int(horizon_hidden)

        self.driver_encoder = TemporalEncoder(driver_dim, driver_hidden, num_layers, driver_heads, dropout)
        self.target_input_embedding = nn.Linear(1, target_embedding)
        self.target_encoder = TemporalEncoder(target_embedding, target_hidden, num_layers, target_heads, dropout)
        self.target_decoder = nn.LSTM(
            input_size=target_embedding,
            hidden_size=target_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.calendar_encoder = nn.Sequential(
            nn.Linear(calendar_dim, calendar_hidden), nn.GELU(), nn.LayerNorm(calendar_hidden)
        )

        ar_fusion_dim = target_hidden + driver_hidden + calendar_hidden
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(ar_fusion_dim),
            nn.Linear(ar_fusion_dim, max(32, ar_fusion_dim)),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(max(32, ar_fusion_dim), 1),
        )

        # Horizon code = normalized index, sqrt(index), sin(pi index), cos(pi index).
        self.horizon_encoder = nn.Sequential(
            nn.Linear(4, horizon_hidden), nn.GELU(), nn.LayerNorm(horizon_hidden)
        )
        direct_fusion_dim = target_hidden + driver_hidden + calendar_hidden + horizon_hidden
        self.direct_head = nn.Sequential(
            nn.LayerNorm(direct_fusion_dim),
            nn.Linear(direct_fusion_dim, max(64, direct_fusion_dim)),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(max(64, direct_fusion_dim), 1),
        )

        self.adversary = nn.Sequential(
            nn.LayerNorm(driver_hidden), nn.Linear(driver_hidden, driver_hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(driver_hidden, summary_dim)
        )

    @staticmethod
    def _target_instance_normalize(y_history: torch.Tensor):
        mean = y_history.mean(dim=1, keepdim=True).detach()
        scale = y_history.var(dim=1, unbiased=False, keepdim=True).add(1e-5).sqrt().detach()
        return (y_history - mean) / scale, TargetInstanceStats(mean=mean, scale=scale)

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
        y_norm, stats = self._target_instance_normalize(y_history)
        emb = self.target_input_embedding(y_norm)
        _, context, hidden, cell = self.target_encoder(emb)
        return context, hidden, cell, stats, y_norm

    def adversary_prediction(self, driver_context: torch.Tensor) -> torch.Tensor:
        return self.adversary(driver_context)

    def _ar_prediction_norm(self, target_states, driver_context, calendar_future):
        h = target_states.shape[1]
        driver = driver_context[:, None, :].expand(-1, h, -1)
        calendar = self.calendar_encoder(calendar_future)
        return self.forecast_head(torch.cat([target_states, driver, calendar], dim=-1))

    def _horizon_features(self, horizon: int, batch: int, device, dtype):
        if horizon <= 1:
            pos = torch.zeros(1, device=device, dtype=dtype)
        else:
            pos = torch.arange(horizon, device=device, dtype=dtype) / float(horizon - 1)
        feat = torch.stack([
            pos,
            torch.sqrt(pos.clamp_min(0.0)),
            torch.sin(math.pi * pos),
            torch.cos(math.pi * pos),
        ], dim=-1)
        return feat[None, :, :].expand(batch, -1, -1)

    def _direct_prediction_norm(self, target_context, driver_context, calendar_future):
        b, h, _ = calendar_future.shape
        target = target_context[:, None, :].expand(-1, h, -1)
        driver = driver_context[:, None, :].expand(-1, h, -1)
        calendar = self.calendar_encoder(calendar_future)
        horizon = self.horizon_encoder(self._horizon_features(h, b, calendar_future.device, calendar_future.dtype))
        return self.direct_head(torch.cat([target, driver, calendar, horizon], dim=-1))

    def _direct_weight_vector(self, horizon: int, device, dtype):
        if self.decoder_mode == "ar":
            w = torch.zeros(horizon, device=device, dtype=dtype)
        elif self.decoder_mode == "hybrid_constant":
            w = torch.full((horizon,), self.direct_weight, device=device, dtype=dtype)
        else:
            if horizon <= 1:
                pos = torch.zeros(1, device=device, dtype=dtype)
            else:
                pos = torch.arange(horizon, device=device, dtype=dtype) / float(horizon - 1)
            w = self.direct_weight_start + (self.direct_weight_end - self.direct_weight_start) * pos
        return w.clamp(0.0, 1.0)[None, :, None]

    def _blend(self, ar_norm, direct_norm):
        w = self._direct_weight_vector(ar_norm.shape[1], ar_norm.device, ar_norm.dtype)
        return (1.0 - w) * ar_norm + w * direct_norm

    def forecast_training(
        self,
        x_history: torch.Tensor,
        y_history: torch.Tensor,
        y_future: torch.Tensor,
        calendar_future: torch.Tensor,
        detach_driver: bool = False,
        rollout_block_size: int = 0,
        rollout_mix: float = 0.0,
    ):
        """Training forecast with optional block rollout exposure.

        Within each block, ground-truth previous targets are used in parallel.
        At block boundaries, the next block can be seeded by a convex mixture of
        the true boundary target and the model's detached prediction.  Setting
        block_size=0 or mix=0 recovers ordinary vectorized teacher forcing.
        """
        rollout_mix = float(rollout_mix)
        if not 0.0 <= rollout_mix <= 1.0:
            raise ValueError("rollout_mix must be in [0,1]")
        driver_context = self.encode_driver(x_history)
        if detach_driver:
            driver_context = driver_context.detach()
        target_context, hidden, cell, stats, y_hist_norm = self.encode_target(y_history)
        future_norm = self._normalize_with_stats(y_future, stats)
        direct_norm = self._direct_prediction_norm(target_context, driver_context, calendar_future)
        horizon = int(y_future.shape[1])

        if rollout_block_size <= 0 or rollout_mix <= 0.0 or rollout_block_size >= horizon:
            decoder_input = torch.cat([y_hist_norm[:, -1:, :], future_norm[:, :-1, :]], dim=1)
            target_states, _ = self.target_decoder(self.target_input_embedding(decoder_input), (hidden, cell))
            ar_norm = self._ar_prediction_norm(target_states, driver_context, calendar_future)
            final_norm = self._blend(ar_norm, direct_norm)
        else:
            bs = int(rollout_block_size)
            ar_parts, final_parts = [], []
            previous_boundary = y_hist_norm[:, -1:, :]
            start = 0
            while start < horizon:
                end = min(start + bs, horizon)
                block_len = end - start
                if block_len == 1:
                    decoder_input = previous_boundary
                else:
                    teacher_rest = future_norm[:, start : end - 1, :]
                    decoder_input = torch.cat([previous_boundary, teacher_rest], dim=1)
                target_states, (hidden, cell) = self.target_decoder(
                    self.target_input_embedding(decoder_input), (hidden, cell)
                )
                ar_block = self._ar_prediction_norm(
                    target_states, driver_context, calendar_future[:, start:end, :]
                )
                direct_block = direct_norm[:, start:end, :]
                # Weight must respect the absolute horizon index, not restart in each block.
                full_w = self._direct_weight_vector(horizon, ar_block.device, ar_block.dtype)[:, start:end, :]
                final_block = (1.0 - full_w) * ar_block + full_w * direct_block
                ar_parts.append(ar_block)
                final_parts.append(final_block)
                if end < horizon:
                    true_boundary = future_norm[:, end - 1 : end, :]
                    model_boundary = final_block[:, -1:, :].detach()
                    previous_boundary = (1.0 - rollout_mix) * true_boundary + rollout_mix * model_boundary
                start = end
            ar_norm = torch.cat(ar_parts, dim=1)
            final_norm = torch.cat(final_parts, dim=1)

        branches = {
            "ar": self._denormalize(ar_norm, stats),
            "direct": self._denormalize(direct_norm, stats),
        }
        return self._denormalize(final_norm, stats), driver_context, branches

    def forecast_teacher_forced(self, x_history, y_history, y_future, calendar_future, detach_driver=False):
        pred, ctx, _ = self.forecast_training(
            x_history, y_history, y_future, calendar_future,
            detach_driver=detach_driver, rollout_block_size=0, rollout_mix=0.0
        )
        return pred, ctx

    def forecast_autoregressive(self, x_history, y_history, calendar_future, detach_driver=False):
        driver_context = self.encode_driver(x_history)
        if detach_driver:
            driver_context = driver_context.detach()
        target_context, hidden, cell, stats, y_hist_norm = self.encode_target(y_history)
        direct_norm = self._direct_prediction_norm(target_context, driver_context, calendar_future)
        full_w = self._direct_weight_vector(calendar_future.shape[1], calendar_future.device, calendar_future.dtype)
        previous = y_hist_norm[:, -1:, :]
        outputs = []
        for h in range(calendar_future.shape[1]):
            state, (hidden, cell) = self.target_decoder(self.target_input_embedding(previous), (hidden, cell))
            ar_step = self._ar_prediction_norm(state, driver_context, calendar_future[:, h:h+1, :])
            direct_step = direct_norm[:, h:h+1, :]
            w = full_w[:, h:h+1, :]
            final_step = (1.0 - w) * ar_step + w * direct_step
            outputs.append(final_step)
            previous = final_step
        pred_norm = torch.cat(outputs, dim=1)
        return self._denormalize(pred_norm, stats), driver_context

    def driver_parameters(self) -> list[nn.Parameter]:
        return list(self.driver_encoder.parameters())

    def forecast_parameters(self) -> list[nn.Parameter]:
        params = []
        modules: Iterable[nn.Module] = (
            self.target_input_embedding, self.target_encoder, self.target_decoder,
            self.calendar_encoder, self.forecast_head, self.horizon_encoder, self.direct_head,
        )
        for module in modules:
            params.extend(module.parameters())
        return params

    def adversary_parameters(self) -> list[nn.Parameter]:
        return list(self.adversary.parameters())

    def non_adversary_parameters(self) -> list[nn.Parameter]:
        return [*self.driver_parameters(), *self.forecast_parameters()]
