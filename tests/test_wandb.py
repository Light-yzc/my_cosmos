import sys
from types import SimpleNamespace

from scripts.train import initialize_wandb


class FakeRun:
    def define_metric(self, *args, **kwargs) -> None:
        del args, kwargs


def test_wandb_run_id_persists_in_checkpoint_mirror(
    tmp_path,
    monkeypatch,
) -> None:
    generated: list[str] = []
    calls: list[dict[str, object]] = []

    def generate_id() -> str:
        value = f"run-{len(generated) + 1}"
        generated.append(value)
        return value

    def init(**kwargs):
        calls.append(kwargs)
        return FakeRun()

    fake = SimpleNamespace(
        util=SimpleNamespace(generate_id=generate_id),
        init=init,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    config = {
        "checkpoint_mirror_dir": str(tmp_path),
        "wandb": {"enabled": True, "project": "test"},
    }
    initialize_wandb({}, config)
    initialize_wandb({}, config)
    assert generated == ["run-1"]
    assert [call["id"] for call in calls] == ["run-1", "run-1"]
    assert (tmp_path / "wandb-run-id.txt").read_text().strip() == "run-1"
