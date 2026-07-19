import pytest

from scripts.train import (
    apply_runtime_overrides,
    create_argument_parser,
    inference_preview_config,
)


def test_smoke_runtime_overrides_are_isolated_from_yaml_defaults() -> None:
    args = create_argument_parser().parse_args(
        [
            "--config",
            "rolling.yaml",
            "--max-steps",
            "8",
            "--rolling-block-size",
            "256",
            "--output-dir",
            "/content/checkpoints_smoke",
            "--checkpoint-mirror-dir",
            "/content/drive/MyDrive/checkpoints_smoke",
        ]
    )
    data = {"rolling_block_size": 2048}
    train = {"max_steps": 500000, "output_dir": "/content/checkpoints"}
    apply_runtime_overrides(args, data, train)
    assert data["rolling_block_size"] == 256
    assert train["max_steps"] == 8
    assert train["output_dir"] == "/content/checkpoints_smoke"
    assert train["checkpoint_mirror_dir"].endswith("checkpoints_smoke")


def test_runtime_overrides_reject_nonpositive_values() -> None:
    args = create_argument_parser().parse_args(["--max-steps", "0"])
    with pytest.raises(ValueError, match="--max-steps must be positive"):
        apply_runtime_overrides(args, {}, {})


def test_wandb_preview_requires_wandb_logging() -> None:
    with pytest.raises(ValueError, match="wandb.enabled"):
        inference_preview_config(
            {
                "wandb": {
                    "enabled": False,
                    "preview": {
                        "enabled": True,
                        "every_steps": 1000,
                        "prompts": ["1girl"],
                    },
                }
            }
        )
