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

