import sys

import pytest
import torch

from my_sd.training.optimizers import build_optimizer


def _config(*, allow_fallback: bool) -> dict[str, object]:
    return {
        "optimizer": "adamw8bit",
        "allow_optimizer_fallback": allow_fallback,
        "learning_rate": 1e-4,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
    }


def test_required_8bit_optimizer_fails_instead_of_allocating_fp32_state(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(RuntimeError, match="8-bit AdamW is required"):
        build_optimizer(torch.nn.Linear(2, 2), _config(allow_fallback=False))


def test_explicit_optimizer_fallback_uses_torch_adamw(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.warns(UserWarning, match="falling back to torch AdamW"):
        optimizer = build_optimizer(
            torch.nn.Linear(2, 2), _config(allow_fallback=True)
        )
    assert isinstance(optimizer, torch.optim.AdamW)


def test_unknown_optimizer_is_rejected() -> None:
    config = _config(allow_fallback=True)
    config["optimizer"] = "mystery"
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(torch.nn.Linear(2, 2), config)
