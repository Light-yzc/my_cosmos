from pathlib import Path

from my_sd.training.checkpoints import (
    AsyncCheckpointMirror,
    checkpoint_is_complete,
    resolve_resume_path,
)


def _checkpoint(root: Path, step: int, *, complete: bool = True) -> Path:
    path = root / f"step-{step:08d}"
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"model")
    if complete:
        (path / "training_state.pt").write_bytes(b"state")
    return path


def test_auto_resume_ignores_partial_and_chooses_newest_across_roots(
    tmp_path,
) -> None:
    local = tmp_path / "local"
    mirror = tmp_path / "mirror"
    _checkpoint(local, 10)
    _checkpoint(local, 30, complete=False)
    newest = _checkpoint(mirror, 20)
    assert resolve_resume_path("auto", local, mirror) == newest


def test_async_checkpoint_mirror_is_complete_and_prunes(tmp_path) -> None:
    local = tmp_path / "local"
    mirror = tmp_path / "mirror"
    first = _checkpoint(local, 1)
    second = _checkpoint(local, 2)
    worker = AsyncCheckpointMirror(
        mirror,
        keep_last=1,
        keep_last_local=1,
    )
    worker.submit(first)
    worker.submit(second)
    worker.close()
    mirrored = mirror / second.name
    assert checkpoint_is_complete(mirrored)
    assert (mirror / "latest.txt").read_text() == second.name
    assert not first.exists()
    assert not (mirror / first.name).exists()
