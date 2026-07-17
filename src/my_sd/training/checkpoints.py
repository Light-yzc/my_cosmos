from __future__ import annotations

import os
import shutil
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

REQUIRED_CHECKPOINT_FILES = ("model.safetensors", "training_state.pt")


def checkpoint_is_complete(path: str | Path) -> bool:
    directory = Path(path)
    return directory.is_dir() and all(
        (directory / filename).is_file()
        for filename in REQUIRED_CHECKPOINT_FILES
    )


def checkpoint_step(path: str | Path) -> int:
    name = Path(path).name
    if not name.startswith("step-"):
        return -1
    try:
        return int(name.removeprefix("step-"))
    except ValueError:
        return -1


def valid_checkpoints(root: str | Path | None) -> list[Path]:
    if root is None:
        return []
    directory = Path(root)
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.glob("step-*")
            if checkpoint_is_complete(path)
        ),
        key=checkpoint_step,
    )


def atomic_write_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, destination)


def update_latest(root: str | Path, checkpoint: str | Path) -> None:
    atomic_write_text(Path(root) / "latest.txt", Path(checkpoint).name)


def resolve_resume_path(
    value: str | None,
    output_dir: str | Path,
    mirror_dir: str | Path | None = None,
) -> Path | None:
    if value is None:
        return None
    if value != "auto":
        path = Path(value)
        path = path.parent if path.is_file() else path
        if not checkpoint_is_complete(path):
            raise FileNotFoundError(
                f"{path} is not a complete checkpoint; expected "
                f"{', '.join(REQUIRED_CHECKPOINT_FILES)}"
            )
        return path

    candidates = valid_checkpoints(output_dir)
    candidates.extend(valid_checkpoints(mirror_dir))
    if not candidates:
        return None
    return max(candidates, key=checkpoint_step)


def prune_checkpoints(
    root: str | Path,
    keep_last: int,
    *,
    protected: Iterable[str | Path] = (),
) -> None:
    if keep_last <= 0:
        return
    protected_paths = {Path(path).resolve() for path in protected}
    checkpoints = valid_checkpoints(root)
    removable = checkpoints[: max(0, len(checkpoints) - keep_last)]
    for checkpoint in removable:
        if checkpoint.resolve() in protected_paths:
            continue
        shutil.rmtree(checkpoint)


def mirror_checkpoint(
    checkpoint_dir: str | Path,
    mirror_dir: str | Path,
    *,
    keep_last: int = 0,
) -> Path:
    source = Path(checkpoint_dir)
    if not checkpoint_is_complete(source):
        raise FileNotFoundError(f"Cannot mirror incomplete checkpoint: {source}")
    root = Path(mirror_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / source.name
    if not checkpoint_is_complete(destination):
        temporary = root / (
            f".{source.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            shutil.copytree(source, temporary)
            if destination.exists():
                shutil.rmtree(destination)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    update_latest(root, destination)
    prune_checkpoints(root, keep_last, protected=(destination,))
    return destination


class AsyncCheckpointMirror:
    """Copies completed local checkpoints to Drive/storage on one worker."""

    def __init__(
        self,
        mirror_dir: str | Path,
        *,
        keep_last: int = 0,
        keep_last_local: int = 0,
    ) -> None:
        self.mirror_dir = Path(mirror_dir)
        self.keep_last = keep_last
        self.keep_last_local = keep_last_local
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: list[tuple[Future[Path], Path]] = []
        self.closed = False

    def _copy_and_prune(self, checkpoint_dir: Path) -> Path:
        mirrored = mirror_checkpoint(
            checkpoint_dir,
            self.mirror_dir,
            keep_last=self.keep_last,
        )
        prune_checkpoints(
            checkpoint_dir.parent,
            self.keep_last_local,
            protected=(checkpoint_dir,),
        )
        return mirrored

    def _reap(self, *, wait: bool) -> None:
        remaining: list[tuple[Future[Path], Path]] = []
        for future, source in self.pending:
            if wait or future.done():
                future.result()
            else:
                remaining.append((future, source))
        self.pending = remaining

    def submit(self, checkpoint_dir: str | Path) -> None:
        if self.closed:
            raise RuntimeError("Checkpoint mirror is closed")
        self._reap(wait=False)
        source = Path(checkpoint_dir)
        future = self.executor.submit(self._copy_and_prune, source)
        self.pending.append((future, source))

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._reap(wait=True)
        finally:
            self.executor.shutdown(wait=True)
            self.closed = True

    def __enter__(self) -> "AsyncCheckpointMirror":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
