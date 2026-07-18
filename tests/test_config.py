from pathlib import Path

from my_sd.config import load_yaml


def test_yaml_extends_deep_merges(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("model:\n  depth: 20\n  width: 1280\n", encoding="utf-8")
    child.write_text(
        "extends: base.yaml\nmodel:\n  depth: 27\ntrain:\n  seed: 1\n",
        encoding="utf-8",
    )
    loaded = load_yaml(child)
    assert loaded["model"] == {"depth": 27, "width": 1280}
    assert loaded["train"]["seed"] == 1


def test_l4_fa2_config_uses_durable_drive_mirror() -> None:
    root = Path(__file__).resolve().parents[1]
    loaded = load_yaml(root / "configs" / "colab_l4_fa2_24gb.yaml")
    assert loaded["model"]["self_attention_backend"] == "flash_attn_2"
    assert loaded["model"]["depth"] == 27
    assert loaded["data"]["resolution_stage"] == "768"
    assert loaded["train"]["precision"] == "bfloat16"
    assert loaded["train"]["parameter_precision"] == "bfloat16"
    assert (
        loaded["train"]["checkpoint_mirror_dir"]
        == "/content/drive/MyDrive/cosmos"
    )


def test_deepghs_l4_config_uses_homogeneous_microbatch_four() -> None:
    root = Path(__file__).resolve().parents[1]
    loaded = load_yaml(root / "configs" / "colab_l4_fa2_deepghs.yaml")
    assert loaded["data"]["batch_size"] == 4
    assert loaded["train"]["gradient_accumulation_steps"] == 4
    assert (
        loaded["data"]["batch_size"]
        * loaded["train"]["gradient_accumulation_steps"]
        == 16
    )
    assert loaded["model"]["gradient_checkpointing"] is False
