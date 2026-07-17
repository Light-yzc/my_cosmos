from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .cosmos_dit import CosmosDiT, CosmosDiTConfig


def initialize_model(
    config: CosmosDiTConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> CosmosDiT:
    """Construct directly in the target dtype/device with initialized parameters."""
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = CosmosDiT(config)
    finally:
        torch.set_default_dtype(old_dtype)
    return model


def resolve_model_weights(path: str | Path) -> Path:
    candidate = Path(path)
    weights = candidate / "model.safetensors" if candidate.is_dir() else candidate
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return weights


def load_model_weights(model: CosmosDiT, path: str | Path) -> None:
    """Stage checkpoint tensors on CPU to avoid a second full GPU model copy."""
    weights = load_file(str(resolve_model_weights(path)), device="cpu")
    try:
        model.load_state_dict(weights, strict=True)
    finally:
        del weights
