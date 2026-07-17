from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .captions import DanbooruCaptioner


def read_shard_list(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        shards = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not shards:
        raise ValueError(f"No shard paths or URLs found in {path}")
    return shards


def _expand_hf_url(source: str) -> str:
    """
    Expands hf://datasets/OWNER/REPO/path/to/shard.tar to a resolve URL.

    Public repositories work without credentials. If HF_TOKEN is set, the
    downloader also sends it for private repositories.
    """
    if not source.startswith("hf://"):
        return source
    parts = source.removeprefix("hf://").split("/")
    if len(parts) < 5 or parts[0] not in {"datasets", "models"}:
        raise ValueError(
            "HF URLs must look like hf://datasets/OWNER/REPO/path/to/file.tar"
        )
    repo_type, owner, repo, *filename = parts
    prefix = "datasets/" if repo_type == "datasets" else ""
    quoted_name = "/".join(urllib.parse.quote(part) for part in filename)
    return f"https://huggingface.co/{prefix}{owner}/{repo}/resolve/main/{quoted_name}"


class AsyncShardPrefetcher:
    """Downloads upcoming tar shards while the caller consumes the current one."""

    def __init__(
        self,
        sources: Sequence[str],
        cache_dir: str | Path,
        *,
        prefetch: int = 2,
        delete_after_use: bool = True,
        retries: int = 4,
        timeout_seconds: int = 120,
        minimum_free_bytes: int = 0,
        max_cache_bytes: int | None = None,
    ) -> None:
        if prefetch < 1:
            raise ValueError("prefetch must be positive")
        if retries < 1:
            raise ValueError("retries must be positive")
        if minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes cannot be negative")
        if max_cache_bytes is not None and max_cache_bytes < 1:
            raise ValueError("max_cache_bytes must be positive")
        self.sources = list(sources)
        self.cache_dir = Path(cache_dir)
        self.prefetch = prefetch
        self.delete_after_use = delete_after_use
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.minimum_free_bytes = minimum_free_bytes
        self.max_cache_bytes = max_cache_bytes
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _response_total_bytes(
        response: Any,
        *,
        existing_bytes: int,
        status: int,
    ) -> int | None:
        content_range = response.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
        content_length = response.headers.get("Content-Length")
        if content_length and content_length.isdigit():
            length = int(content_length)
            return existing_bytes + length if status == 206 else length
        return None

    def _check_disk_budget(self, additional_bytes: int | None) -> None:
        usage = shutil.disk_usage(self.cache_dir)
        required = self.minimum_free_bytes + max(additional_bytes or 0, 0)
        if usage.free < required:
            raise OSError(
                f"Insufficient free space in {self.cache_dir}: "
                f"{usage.free:,} bytes free, {required:,} required"
            )
        if self.max_cache_bytes is not None:
            current = sum(
                path.stat().st_size
                for path in self.cache_dir.iterdir()
                if path.is_file()
            )
            projected = current + max(additional_bytes or 0, 0)
            if projected > self.max_cache_bytes:
                raise OSError(
                    f"Shard cache budget exceeded: {projected:,} > "
                    f"{self.max_cache_bytes:,} bytes"
                )

    def _download_once(
        self,
        url: str,
        temporary: Path,
    ) -> int | None:
        existing = temporary.stat().st_size if temporary.is_file() else 0
        headers = {"User-Agent": "cosmos-anime-colab/0.1"}
        token = os.environ.get("HF_TOKEN")
        if token and "huggingface.co" in urllib.parse.urlparse(url).netloc:
            headers["Authorization"] = f"Bearer {token}"
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(
            request, timeout=self.timeout_seconds
        ) as response:
            status = int(getattr(response, "status", response.getcode()))
            append = existing > 0 and status == 206
            if not append:
                existing = 0
            total = self._response_total_bytes(
                response,
                existing_bytes=existing,
                status=status,
            )
            remaining = total - existing if total is not None else None
            self._check_disk_budget(remaining)
            mode = "ab" if append else "wb"
            with temporary.open(mode) as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        return total

    def _download(self, source: str) -> tuple[Path, bool]:
        local = Path(source)
        if local.is_file():
            return local.resolve(), False

        url = _expand_hf_url(source)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise FileNotFoundError(
                f"Shard is neither a local file nor HTTP URL: {source}"
            )
        basename = Path(parsed.path).name or "shard.tar"
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        destination = self.cache_dir / f"{digest}-{basename}"
        if destination.is_file():
            return destination, True

        temporary = destination.with_suffix(destination.suffix + ".part")
        last_error: Exception | None = None
        expected_total: int | None = None
        for attempt in range(self.retries):
            try:
                expected_total = self._download_once(url, temporary)
                actual_size = temporary.stat().st_size
                if expected_total is not None and actual_size != expected_total:
                    raise OSError(
                        f"Incomplete shard {source}: got {actual_size:,} of "
                        f"{expected_total:,} bytes"
                    )
                break
            except Exception as error:
                last_error = error
                if attempt + 1 >= self.retries:
                    raise
                time.sleep(min(2**attempt, 8))
        else:
            assert last_error is not None
            raise last_error

        temporary.replace(destination)
        return destination, True

    def __iter__(self) -> Iterator[Path]:
        with ThreadPoolExecutor(max_workers=self.prefetch) as executor:
            futures: list[Future[tuple[Path, bool]]] = []
            next_index = 0
            while next_index < min(self.prefetch, len(self.sources)):
                futures.append(executor.submit(self._download, self.sources[next_index]))
                next_index += 1

            while futures:
                path, downloaded = futures.pop(0).result()
                if next_index < len(self.sources):
                    futures.append(
                        executor.submit(self._download, self.sources[next_index])
                    )
                    next_index += 1
                try:
                    yield path
                finally:
                    if downloaded and self.delete_after_use and path.is_file():
                        path.unlink()


def _sample_key(name: str) -> tuple[str, str] | None:
    normalized = name.replace("\\", "/")
    if normalized.endswith(".latent.npy"):
        return normalized[: -len(".latent.npy")], "latent"
    if normalized.endswith(".json"):
        return normalized[: -len(".json")], "metadata"
    return None


def iter_latent_tar(path: str | Path) -> Iterator[dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {}
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parsed = _sample_key(member.name)
            if parsed is None:
                continue
            key, kind = parsed
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            sample = pending.setdefault(key, {"sample_id": key})
            if kind == "latent":
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                sample["latent"] = torch.from_numpy(array)
            else:
                sample["metadata"] = json.loads(payload.decode("utf-8"))
            if "latent" in sample and "metadata" in sample:
                yield pending.pop(key)
    if pending:
        missing = ", ".join(list(pending)[:3])
        raise ValueError(f"Incomplete samples in {path}: {missing}")


def buffered_shuffle(
    samples: Iterable[dict[str, Any]],
    *,
    buffer_size: int,
    rng: random.Random,
) -> Iterator[dict[str, Any]]:
    if buffer_size < 1:
        yield from samples
        return
    buffer: list[dict[str, Any]] = []
    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= buffer_size:
            yield buffer.pop(rng.randrange(len(buffer)))
    rng.shuffle(buffer)
    yield from buffer


class StreamingLatentDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        shard_sources: Sequence[str],
        *,
        cache_dir: str | Path = "/content/shard_cache",
        captioner: DanbooruCaptioner | None = None,
        prefetch_shards: int = 2,
        sample_shuffle_buffer: int = 256,
        delete_after_use: bool = True,
        shuffle_shards: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not shard_sources:
            raise ValueError("At least one latent tar shard is required")
        self.shard_sources = list(shard_sources)
        self.cache_dir = Path(cache_dir)
        self.captioner = captioner or DanbooruCaptioner()
        self.prefetch_shards = prefetch_shards
        self.sample_shuffle_buffer = sample_shuffle_buffer
        self.delete_after_use = delete_after_use
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.epoch = 0
        self.resume_epoch: int | None = None
        self.resume_shard_index = 0
        self.resume_sample_index = -1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_resume_cursor(
        self,
        *,
        epoch: int,
        shard_index: int,
        sample_index: int,
    ) -> None:
        self.resume_epoch = epoch
        self.resume_shard_index = shard_index
        self.resume_sample_index = sample_index

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        epoch = self.epoch
        rng = random.Random(self.seed + epoch * 1009 + worker_id)
        sources = list(self.shard_sources)
        if self.shuffle_shards:
            rng.shuffle(sources)
        sources = sources[worker_id::worker_count]
        start_shard = (
            self.resume_shard_index
            if self.resume_epoch == epoch and worker_id == 0
            else 0
        )
        sources = sources[start_shard:]
        prefetcher = AsyncShardPrefetcher(
            sources,
            self.cache_dir / f"worker-{worker_id}",
            prefetch=self.prefetch_shards,
            delete_after_use=self.delete_after_use,
        )
        global_sample_index = 0
        for shard_offset, shard_path in enumerate(prefetcher):
            shard_index = start_shard + shard_offset
            sample_rng = random.Random(
                self.seed + epoch * 1_000_003 + shard_index * 9176 + worker_id
            )
            caption_rng = random.Random(
                self.seed + epoch * 2_000_003 + shard_index * 7919 + worker_id
            )
            samples = buffered_shuffle(
                iter_latent_tar(shard_path),
                buffer_size=self.sample_shuffle_buffer,
                rng=sample_rng,
            )
            for sample_index, sample in enumerate(samples):
                metadata = sample["metadata"]
                if any(
                    key in metadata
                    for key in (
                        "general_tags",
                        "tag_string_general",
                        "character_tags",
                        "tag_string_character",
                    )
                ):
                    caption = self.captioner.compose(metadata, caption_rng)
                else:
                    caption = str(metadata.get("caption", ""))
                if (
                    self.resume_epoch == epoch
                    and shard_index == self.resume_shard_index
                    and sample_index <= self.resume_sample_index
                ):
                    continue
                yield {
                    "latent": sample["latent"],
                    "caption": caption,
                    "bucket": str(metadata["bucket"]),
                    "index": global_sample_index,
                    "sample_id": sample["sample_id"],
                    "stream_epoch": epoch,
                    "source_shard_index": shard_index,
                    "source_sample_index": sample_index,
                }
                global_sample_index += 1
        self.resume_epoch = None


class LatentTarWriter:
    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination)
        self.temporary = self.destination.with_suffix(self.destination.suffix + ".partial")
        self.archive: tarfile.TarFile | None = None
        self.sample_count = 0

    def __enter__(self) -> "LatentTarWriter":
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.archive = tarfile.open(self.temporary, mode="w")
        return self

    def _add(self, name: str, payload: bytes) -> None:
        if self.archive is None:
            raise RuntimeError("LatentTarWriter is not open")
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        info.mtime = 0
        info.mode = 0o644
        self.archive.addfile(info, io.BytesIO(payload))

    def add(
        self,
        sample_id: str,
        latent: torch.Tensor,
        metadata: dict[str, Any],
    ) -> None:
        array_buffer = io.BytesIO()
        np.save(
            array_buffer,
            latent.detach().to(device="cpu", dtype=torch.float16).numpy(),
            allow_pickle=False,
        )
        self._add(f"{sample_id}.latent.npy", array_buffer.getvalue())
        self._add(
            f"{sample_id}.json",
            json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
        )
        self.sample_count += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.archive is not None:
            self.archive.close()
        if exc_type is None:
            self.temporary.replace(self.destination)
        elif self.temporary.is_file():
            self.temporary.unlink()
