import pytest
import torch

from my_sd.training.precision import (
    grad_scaler_enabled,
    precision_dtype,
    training_dtypes,
)


def test_fp16_compute_defaults_to_fp32_master_parameters() -> None:
    compute, parameters = training_dtypes({"precision": "float16"})
    assert compute == torch.float16
    assert parameters == torch.float32
    assert grad_scaler_enabled(compute, parameters)


def test_bf16_keeps_bf16_parameters_without_scaler() -> None:
    compute, parameters = training_dtypes({"precision": "bfloat16"})
    assert compute == torch.bfloat16
    assert parameters == torch.bfloat16
    assert not grad_scaler_enabled(compute, parameters)


def test_explicit_fp16_parameters_disable_incompatible_scaler() -> None:
    compute, parameters = training_dtypes(
        {"precision": "float16", "parameter_precision": "float16"}
    )
    assert parameters == torch.float16
    assert not grad_scaler_enabled(compute, parameters)


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        precision_dtype("int8")
