import json
from pathlib import Path


def test_colab_notebook_is_valid_and_contains_deployment_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "colab_rolling_train.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    required = (
        "drive.mount",
        "HF_TOKEN",
        "Wan2.2_VAE.pth",
        "list_hf_shards.py",
        "colab_preflight.py",
        '"--resume", "auto"',
    )
    for value in required:
        assert value in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}#{cell['id']}", "exec")
