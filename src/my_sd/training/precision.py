from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def precision_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return values[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported precision: {name}") from error


def training_dtypes(config: Mapping[str, Any]) -> tuple[torch.dtype, torch.dtype]:
    """Return (compute dtype, parameter dtype) for single-GPU AMP training.

    FP16 compute defaults to FP32 master parameters because GradScaler cannot
    unscale gradients stored directly in FP16 parameters. BF16 does not need
    loss scaling and can safely keep BF16 parameters to save memory.
    """
    compute = precision_dtype(str(config.get("precision", "bfloat16")))
    configured = config.get("parameter_precision")
    if configured is None:
        parameters = torch.float32 if compute == torch.float16 else compute
    else:
        parameters = precision_dtype(str(configured))
    return compute, parameters


def grad_scaler_enabled(
    compute_dtype: torch.dtype,
    parameter_dtype: torch.dtype,
) -> bool:
    return compute_dtype == torch.float16 and parameter_dtype != torch.float16
