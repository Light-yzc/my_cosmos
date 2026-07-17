from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    train_config: dict[str, Any],
) -> torch.optim.Optimizer:
    name = str(train_config.get("optimizer", "adamw")).lower()
    kwargs = {
        "lr": float(train_config["learning_rate"]),
        "betas": tuple(train_config.get("betas", (0.9, 0.95))),
        "weight_decay": float(train_config.get("weight_decay", 0.1)),
    }
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(model.parameters(), **kwargs)
        except (ImportError, RuntimeError) as error:
            if not bool(train_config.get("allow_optimizer_fallback", True)):
                raise RuntimeError(
                    "8-bit AdamW is required by this configuration, but "
                    f"bitsandbytes could not initialize: {error}"
                ) from error
            warnings.warn(
                f"8-bit AdamW unavailable ({error}); falling back to torch AdamW. "
                "This uses roughly 5-7 GB more memory.",
                stacklevel=2,
            )
    elif name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.AdamW(model.parameters(), foreach=False, **kwargs)
