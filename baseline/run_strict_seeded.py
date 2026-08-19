#!/usr/bin/env python3
"""Deterministic execution wrapper for the frozen seeded baseline runner.

The paper-facing baseline implementation and hyperparameters remain in run_seeded.py.
This wrapper only configures deterministic PyTorch/CUDA execution before delegating
argument parsing and training to that frozen runner.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

# Must be set before CUDA/cuBLAS is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import torch


def configure_strict_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None:
        for name, value in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            fn = getattr(cuda_backend, name, None)
            if callable(fn):
                fn(value)


def main() -> None:
    configure_strict_determinism()
    runpy.run_path(str(Path(__file__).with_name("run_seeded.py")), run_name="__main__")


if __name__ == "__main__":
    main()
