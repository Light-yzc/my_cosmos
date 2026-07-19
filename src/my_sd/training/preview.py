from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class InferencePreviewConfig:
    """Fixed-seed inference settings for tracking training progress."""

    enabled: bool = False
    every_steps: int = 0
    prompts: tuple[str, ...] = ()
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 16
    guidance_scale: float = 5.0
    solver: str = "euler"
    seed: int = 3407

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "InferencePreviewConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise TypeError("train.wandb.preview must be a mapping")
        raw_prompts = values.get("prompts", ())
        if isinstance(raw_prompts, str) or not isinstance(raw_prompts, Sequence):
            raise TypeError("train.wandb.preview.prompts must be a YAML list")
        config = cls(
            enabled=bool(values.get("enabled", False)),
            every_steps=int(values.get("every_steps", 0)),
            prompts=tuple(str(prompt).strip() for prompt in raw_prompts),
            negative_prompt=str(values.get("negative_prompt", "")),
            width=int(values.get("width", 512)),
            height=int(values.get("height", 512)),
            steps=int(values.get("steps", 16)),
            guidance_scale=float(values.get("guidance_scale", 5.0)),
            solver=str(values.get("solver", "euler")),
            seed=int(values.get("seed", 3407)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.every_steps < 1:
            raise ValueError("train.wandb.preview.every_steps must be positive")
        if not self.prompts or any(not prompt for prompt in self.prompts):
            raise ValueError(
                "train.wandb.preview.prompts must contain at least one prompt"
            )
        if self.width < 32 or self.height < 32:
            raise ValueError("preview width and height must be at least 32")
        if self.width % 32 or self.height % 32:
            raise ValueError("preview width and height must be divisible by 32")
        if self.steps < 1:
            raise ValueError("preview steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("preview guidance_scale cannot be negative")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("preview solver must be 'euler' or 'heun'")

    def is_due(self, step: int) -> bool:
        return self.enabled and step > 0 and step % self.every_steps == 0


@dataclass(frozen=True, slots=True)
class InferencePreview:
    prompt: str
    path: Path


PreviewRunner = Callable[..., subprocess.CompletedProcess[Any]]


def run_inference_previews(
    config: InferencePreviewConfig,
    *,
    training_config_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    optimizer_step: int,
    sample_script: str | Path,
    project_root: str | Path,
    python_executable: str = sys.executable,
    runner: PreviewRunner = subprocess.run,
) -> list[InferencePreview]:
    """Run the normal sampling CLI in an isolated process for each fixed prompt."""
    config.validate()
    if not config.enabled:
        return []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    previews: list[InferencePreview] = []
    for index, prompt in enumerate(config.prompts):
        output = destination / (
            f"step-{optimizer_step:08d}-prompt-{index + 1:02d}.png"
        )
        command = [
            python_executable,
            str(sample_script),
            "--config",
            str(training_config_path),
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            prompt,
            "--negative-prompt",
            config.negative_prompt,
            "--width",
            str(config.width),
            "--height",
            str(config.height),
            "--steps",
            str(config.steps),
            "--guidance-scale",
            str(config.guidance_scale),
            "--solver",
            config.solver,
            "--seed",
            str(config.seed + index),
            "--output",
            str(output),
        ]
        runner(command, cwd=str(project_root), check=True)
        if not output.is_file():
            raise RuntimeError(f"Inference preview did not create {output}")
        previews.append(InferencePreview(prompt=prompt, path=output))
    return previews
