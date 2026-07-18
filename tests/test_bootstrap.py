from pathlib import Path

from my_sd.config import load_yaml
from scripts.bootstrap_deepghs_colab import (
    training_command,
    write_runtime_config,
)


def test_deepghs_bootstrap_always_auto_resumes() -> None:
    command = training_command("configs/example.yaml")
    assert command[-4:] == (
        "--config",
        "configs/example.yaml",
        "--resume",
        "auto",
    )


def test_smoke_runtime_config_is_isolated(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "data:\n  metadata_index_dir: old\n"
        "train:\n  output_dir: old\n  max_steps: 500000\n",
        encoding="utf-8",
    )
    runtime = write_runtime_config(
        base,
        drive_root=tmp_path / "drive",
        output=tmp_path / "runtime.yaml",
        smoke_steps=8,
    )
    loaded = load_yaml(runtime)
    assert Path(loaded["data"]["metadata_index_dir"]) == (
        tmp_path / "drive" / "deepghs_metadata"
    )
    assert loaded["train"]["output_dir"] == "/content/checkpoints_l4_fa2_smoke"
    assert Path(loaded["train"]["checkpoint_mirror_dir"]) == (
        tmp_path / "drive" / "smoke"
    )
    assert loaded["train"]["max_steps"] == 8
