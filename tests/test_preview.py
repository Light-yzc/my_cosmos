import subprocess
from pathlib import Path

import pytest

from my_sd.training.preview import (
    InferencePreviewConfig,
    run_inference_previews,
)


def test_preview_config_requires_fixed_prompts_and_valid_geometry() -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        InferencePreviewConfig.from_mapping(
            {"enabled": True, "every_steps": 1000}
        )
    with pytest.raises(ValueError, match="divisible by 32"):
        InferencePreviewConfig.from_mapping(
            {
                "enabled": True,
                "every_steps": 1000,
                "prompts": ["1girl"],
                "width": 500,
            }
        )


def test_preview_schedule_uses_optimizer_steps() -> None:
    config = InferencePreviewConfig.from_mapping(
        {
            "enabled": True,
            "every_steps": 1000,
            "prompts": ["1girl"],
        }
    )
    assert not config.is_due(999)
    assert config.is_due(1000)


def test_preview_runs_normal_sampling_cli_with_fixed_seed(tmp_path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0)

    config = InferencePreviewConfig.from_mapping(
        {
            "enabled": True,
            "every_steps": 100,
            "prompts": ["first prompt", "second prompt"],
            "negative_prompt": "blurry",
            "width": 512,
            "height": 512,
            "steps": 12,
            "solver": "euler",
            "seed": 7,
        }
    )
    previews = run_inference_previews(
        config,
        training_config_path="runtime.yaml",
        checkpoint="step-00000100",
        output_dir=tmp_path,
        optimizer_step=100,
        sample_script="scripts/sample.py",
        project_root="/repo",
        python_executable="python",
        runner=fake_runner,
    )

    assert [preview.prompt for preview in previews] == [
        "first prompt",
        "second prompt",
    ]
    assert len(calls) == 2
    first, kwargs = calls[0]
    assert first[:2] == ["python", "scripts/sample.py"]
    assert first[first.index("--checkpoint") + 1] == "step-00000100"
    assert first[first.index("--seed") + 1] == "7"
    assert calls[1][0][calls[1][0].index("--seed") + 1] == "8"
    assert kwargs == {"cwd": "/repo", "check": True}
