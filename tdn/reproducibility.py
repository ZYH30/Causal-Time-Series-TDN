"""Reproducibility controls for the frozen TDN Weather experiments."""
from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass(frozen=True)
class ReproducibilityState:
    mode: str
    seed: int
    python: str
    torch: str
    cuda_runtime: str | None
    cudnn: int | None
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_tf32: bool
    cudnn_tf32: bool
    cublas_workspace_config: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_reproducibility(seed: int, mode: str) -> ReproducibilityState:
    """Configure either the original frozen seed contract or strict determinism.

    ``legacy`` preserves the original v1.5.1 behavior.
    ``strict`` additionally requests deterministic CUDA algorithms, disables
    TF32, and forces the math SDPA backend to remove kernel-dispatch ambiguity.
    The launcher must set CUBLAS_WORKSPACE_CONFIG before Python starts.
    """
    if mode not in {"legacy", "strict"}:
        raise ValueError(f"Unknown reproducibility mode: {mode}")

    _set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if mode == "strict":
        required = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if required not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "Strict mode requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) "
                "to be set before the Python process starts."
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        # MultiheadAttention may dispatch through scaled-dot-product attention.
        # Restrict to the math backend for a single deterministic execution path.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    else:
        # Match the original v1.5.1 seed contract exactly.
        torch.use_deterministic_algorithms(False)

    return ReproducibilityState(
        mode=mode,
        seed=int(seed),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cudnn=(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cuda_matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Return an explicit CPU generator for deterministic train shuffling."""
    return torch.Generator(device="cpu").manual_seed(int(seed))
