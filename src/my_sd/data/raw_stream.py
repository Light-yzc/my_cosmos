from __future__ import annotations

import io
import json
import random
import sys
import tarfile
import threading
import time
from queue import Empty, Full, Queue
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from my_sd.autoencoders import WanImageVAE, WanVAEConfig

from .buckets import (
    DEFAULT_BUCKETS,
    HIGH_RES_BUCKETS,
    LOW_RES_BUCKETS,
    ResolutionBucket,
    choose_bucket,
    cover_resize_and_center_crop,
)
from .captions import DanbooruCaptioner
from .deepghs_metadata import DeepGHSMetadataIndex
from .tar_stream import AsyncShardPrefetcher

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RATING_ALIASES = {
    "g": "g",
    "general": "g",
    "safe": "g",
    "s": "s",
    "sensitive": "s",
    "suggestive": "s",
    "q": "q",
    "questionable": "q",
    "e": "e",
    "explicit": "e",
}


class ImageEncoder(Protocol):
    def encode_images(self, images: Tensor) -> Tensor: ...

    def move_to(self, device: str, dtype: str | None = None) -> None: ...

    def offload_to_cpu(self) -> None: ...


@dataclass(slots=True)
class PreparedImage:
    source_shard_index: int
    source_sample_index: int
    sample_id: str
    pixels: Tensor
    metadata: dict[str, Any]
    bucket: str


@dataclass(slots=True)
class EncodedImage:
    source_shard_index: int
    source_sample_index: int
    sample_id: str
    latent: Tensor
    metadata: dict[str, Any]
    bucket: str


def raw_member_key(name: str) -> tuple[str, str] | None:
    path = Path(name.replace("\\", "/"))
    suffix = path.suffix.lower()
    key = str(path.with_suffix("")).replace("\\", "/")
    if suffix in IMAGE_EXTENSIONS:
        return key, "image"
    if suffix == ".json":
        return key, "json"
    if suffix == ".txt":
        return key, "text"
    return None


def has_caption_metadata(metadata: Mapping[str, object]) -> bool:
    keys = (
        "caption",
        "general_tags",
        "tag_string_general",
        "character_tags",
        "tag_string_character",
        "copyright_tags",
        "tag_string_copyright",
        "artist_tags",
        "tag_string_artist",
    )
    return any(metadata.get(key) for key in keys)


def iter_raw_tar(
    path: str | Path,
    *,
    require_metadata: bool = True,
    external_metadata: Mapping[str, Mapping[str, object]] | None = None,
    progress_label: str | None = None,
    progress_state: dict[str, int] | None = None,
) -> Iterator[tuple[str, Image.Image, dict[str, Any]]]:
    """
    Reads WebDataset-style image + JSON/TXT samples.

    The current implementation builds the member index for one shard. This is
    bounded for the recommended roughly 1.6 GB/25k-image shards and allows
    sidecars to appear before or after their image. It must not be used on one
    monolithic multi-million-member tar.
    """
    scan_started = time.monotonic()
    if progress_label:
        print(
            f"[data] {progress_label}: scanning tar index...",
            file=sys.stderr,
            flush=True,
        )
    with tarfile.open(path, mode="r:*") as archive:
        grouped: dict[str, dict[str, tarfile.TarInfo]] = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parsed = raw_member_key(member.name)
            if parsed is None:
                continue
            key, kind = parsed
            grouped.setdefault(key, {})[kind] = member
        if progress_state is not None:
            progress_state["total"] = len(grouped)
            progress_state["scanned"] = 0
            progress_state["decoded"] = 0
        if progress_label:
            print(
                f"[data] {progress_label}: indexed {len(grouped):,} members "
                f"in {time.monotonic() - scan_started:.1f}s; decoding images...",
                file=sys.stderr,
                flush=True,
            )

        for key, members in grouped.items():
            if progress_state is not None:
                progress_state["scanned"] += 1
            image_member = members.get("image")
            if image_member is None:
                continue
            metadata: dict[str, Any] = {}
            if "json" in members:
                sidecar = archive.extractfile(members["json"])
                if sidecar is not None:
                    loaded = json.loads(sidecar.read().decode("utf-8"))
                    if isinstance(loaded, dict):
                        metadata.update(loaded)
            elif "text" in members:
                sidecar = archive.extractfile(members["text"])
                if sidecar is not None:
                    metadata["general_tags"] = (
                        sidecar.read().decode("utf-8").strip()
                    )
            if external_metadata is not None:
                sample_id = Path(key).name
                indexed = external_metadata.get(sample_id)
                if indexed:
                    metadata.update(indexed)
            if require_metadata and not has_caption_metadata(metadata):
                continue
            image_file = archive.extractfile(image_member)
            if image_file is None:
                continue
            with Image.open(io.BytesIO(image_file.read())) as opened:
                image = opened.convert("RGB")
            if progress_state is not None:
                progress_state["decoded"] += 1
            yield key, image, metadata


def _threaded_prefetch(
    iterable: Iterator[tuple[str, Image.Image, dict[str, Any]]],
    *,
    capacity: int,
) -> Iterator[tuple[str, Image.Image, dict[str, Any]]]:
    """Decode upcoming tar images on CPU while the main thread runs the VAE."""
    if capacity < 1:
        yield from iterable
        return
    queue: Queue[object] = Queue(maxsize=capacity)
    sentinel = object()
    stopped = threading.Event()

    def put(value: object) -> bool:
        while not stopped.is_set():
            try:
                queue.put(value, timeout=0.2)
                return True
            except Full:
                continue
        return False

    def produce() -> None:
        try:
            for value in iterable:
                if not put(value):
                    return
        except BaseException as error:
            put(error)
        finally:
            put(sentinel)

    thread = threading.Thread(
        target=produce,
        name="raw-image-prefetch",
        daemon=True,
    )
    thread.start()
    try:
        while True:
            try:
                value = queue.get(timeout=0.5)
            except Empty:
                if not thread.is_alive():
                    return
                continue
            if value is sentinel:
                return
            if isinstance(value, BaseException):
                raise value
            yield value  # type: ignore[misc]
    finally:
        stopped.set()
        thread.join(timeout=2)


def prepare_image(
    image: Image.Image,
    bucket: ResolutionBucket,
) -> tuple[Tensor, tuple[int, int, int, int]]:
    resize_size, crop_box = cover_resize_and_center_crop(
        image.width, image.height, bucket
    )
    image = image.resize(resize_size, Image.Resampling.LANCZOS).crop(crop_box)
    array = np.array(image, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)
    return tensor, crop_box


def buckets_for_stage(stage: str) -> tuple[ResolutionBucket, ...]:
    values = {
        "512": LOW_RES_BUCKETS,
        "768": DEFAULT_BUCKETS,
        "1024": HIGH_RES_BUCKETS,
    }
    try:
        return values[str(stage)]
    except KeyError as error:
        raise ValueError(f"Unknown resolution stage: {stage}") from error


def _normalized_rating(value: object) -> str:
    return RATING_ALIASES.get(str(value).strip().lower(), "")


class RollingWanDataset(IterableDataset[dict[str, Any]]):
    """
    Alternates between batched Wan encoding and DiT consumption in one process.

    While the caller trains on an encoded CPU block, AsyncShardPrefetcher keeps
    downloading the next raw tar. The Wan encoder is on GPU only while building
    a block, then moves back to CPU before the first sample is yielded.
    """

    def __init__(
        self,
        shard_sources: Sequence[str],
        *,
        wan_config: WanVAEConfig | None = None,
        encoder_factory: Callable[[], ImageEncoder] | None = None,
        cache_dir: str | Path = "/content/raw_cache",
        captioner: DanbooruCaptioner | None = None,
        resolution_stage: str = "512",
        prefetch_shards: int = 1,
        block_size: int = 2048,
        encode_batch_size: int = 4,
        decode_prefetch: int = 16,
        accumulation_multiple: int = 1,
        max_upscale: float = 1.25,
        allowed_ratings: Sequence[str] | None = None,
        require_metadata: bool = True,
        metadata_index_dir: str | Path | None = None,
        delete_after_use: bool = True,
        shuffle_shards: bool = True,
        download_retries: int = 4,
        download_timeout_seconds: int = 120,
        minimum_free_bytes: int = 0,
        max_cache_bytes: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not shard_sources:
            raise ValueError("At least one raw tar shard is required")
        if (wan_config is None) == (encoder_factory is None):
            raise ValueError("Provide exactly one of wan_config or encoder_factory")
        if block_size < 1 or encode_batch_size < 1:
            raise ValueError("block_size and encode_batch_size must be positive")
        if prefetch_shards != 1:
            raise ValueError(
                "rolling raw mode requires prefetch_shards=1 to bound disk usage"
            )
        if accumulation_multiple < 1:
            raise ValueError("accumulation_multiple must be positive")
        if block_size % accumulation_multiple:
            raise ValueError(
                "rolling block_size must be divisible by gradient accumulation"
            )
        if max_upscale <= 0:
            raise ValueError("max_upscale must be positive")
        if decode_prefetch < 0:
            raise ValueError("decode_prefetch cannot be negative")

        self.shard_sources = list(shard_sources)
        self.wan_config = wan_config
        self.encoder_factory = encoder_factory
        self.cache_dir = Path(cache_dir)
        self.captioner = captioner or DanbooruCaptioner()
        self.buckets = buckets_for_stage(resolution_stage)
        self.prefetch_shards = prefetch_shards
        self.block_size = block_size
        self.encode_batch_size = encode_batch_size
        self._effective_encode_batch_size = encode_batch_size
        self.decode_prefetch = decode_prefetch
        self.accumulation_multiple = accumulation_multiple
        self.max_upscale = max_upscale
        self.allowed_ratings = (
            {
                normalized
                for value in allowed_ratings
                if (normalized := _normalized_rating(value))
            }
            if allowed_ratings
            else None
        )
        self.require_metadata = require_metadata
        self.metadata_index = (
            DeepGHSMetadataIndex(metadata_index_dir)
            if metadata_index_dir is not None
            else None
        )
        self.delete_after_use = delete_after_use
        self.shuffle_shards = shuffle_shards
        self.download_retries = download_retries
        self.download_timeout_seconds = download_timeout_seconds
        self.minimum_free_bytes = minimum_free_bytes
        self.max_cache_bytes = max_cache_bytes
        self.seed = seed
        self.epoch = 0
        self.resume_epoch: int | None = None
        self.resume_shard_index = 0
        self.resume_sample_index = -1
        self._encoder: ImageEncoder | None = None

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

    def _get_encoder(self) -> ImageEncoder:
        if self._encoder is None:
            if self.encoder_factory is not None:
                self._encoder = self.encoder_factory()
            else:
                assert self.wan_config is not None
                cpu_config = replace(self.wan_config, device="cpu")
                self._encoder = WanImageVAE(cpu_config)
        return self._encoder

    def _move_encoder_to_compute(self) -> ImageEncoder:
        encoder = self._get_encoder()
        device = self.wan_config.device if self.wan_config is not None else "cpu"
        dtype = self.wan_config.dtype if self.wan_config is not None else None
        encoder.move_to(device, dtype)
        return encoder

    @staticmethod
    def _offload_encoder(encoder: ImageEncoder) -> None:
        encoder.offload_to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode_batch(
        self,
        encoder: ImageEncoder,
        items: list[PreparedImage],
    ) -> list[EncodedImage]:
        if not items:
            return []
        if len(items) > self._effective_encode_batch_size:
            encoded: list[EncodedImage] = []
            for start in range(0, len(items), self._effective_encode_batch_size):
                encoded.extend(
                    self._encode_batch(
                        encoder,
                        items[start : start + self._effective_encode_batch_size],
                    )
                )
            return encoded
        pixels = torch.stack([item.pixels for item in items])
        if torch.cuda.is_available():
            pixels = pixels.pin_memory()
        try:
            latents = encoder.encode_images(pixels)
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(items) == 1:
                raise
            midpoint = len(items) // 2
            self._effective_encode_batch_size = min(
                self._effective_encode_batch_size,
                midpoint,
            )
            print(
                f"[encode] OOM at batch={len(items)}; retrying as "
                f"{midpoint}+{len(items) - midpoint}; future batches use "
                f"{self._effective_encode_batch_size}",
                file=sys.stderr,
                flush=True,
            )
            return [
                *self._encode_batch(encoder, items[:midpoint]),
                *self._encode_batch(encoder, items[midpoint:]),
            ]
        if len(latents) != len(items):
            raise RuntimeError(
                f"Wan encoder returned {len(latents)} latents for {len(items)} images"
            )
        encoded: list[EncodedImage] = []
        for item, latent in zip(items, latents, strict=True):
            encoded.append(
                EncodedImage(
                    source_shard_index=item.source_shard_index,
                    source_sample_index=item.source_sample_index,
                    sample_id=item.sample_id,
                    latent=latent.detach()
                    .to(device="cpu", dtype=torch.float16)
                    .contiguous(),
                    metadata=item.metadata,
                    bucket=item.bucket,
                )
            )
        return encoded

    def _caption(self, item: EncodedImage) -> str:
        caption_seed = (
            self.seed
            + self.epoch * 2_000_003
            + item.source_shard_index * 7919
            + item.source_sample_index * 104729
        )
        rng = random.Random(caption_seed)
        if has_caption_metadata(item.metadata):
            return self.captioner.compose(item.metadata, rng)
        return str(item.metadata.get("caption", ""))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "RollingWanDataset performs GPU encoding and requires num_workers=0"
            )

        epoch = self.epoch
        shard_rng = random.Random(self.seed + epoch * 1009)
        sources = list(self.shard_sources)
        if self.shuffle_shards:
            shard_rng.shuffle(sources)
        start_shard = (
            self.resume_shard_index if self.resume_epoch == epoch else 0
        )
        sources = sources[start_shard:]
        prefetcher = AsyncShardPrefetcher(
            sources,
            self.cache_dir,
            prefetch=self.prefetch_shards,
            delete_after_use=self.delete_after_use,
            retries=self.download_retries,
            timeout_seconds=self.download_timeout_seconds,
            minimum_free_bytes=self.minimum_free_bytes,
            max_cache_bytes=self.max_cache_bytes,
        )

        encoder = self._move_encoder_to_compute()
        pending_by_bucket: dict[str, list[PreparedImage]] = {}
        encoded_buffer: list[EncodedImage] = []
        global_index = 0
        encode_started = time.monotonic()
        encoded_since_report = 0
        last_encode_report = encode_started

        def flush_bucket(bucket_key: str) -> None:
            nonlocal encoded_since_report, last_encode_report
            items = pending_by_bucket.get(bucket_key, [])
            if not items:
                return
            pending_by_bucket[bucket_key] = []
            encoded_buffer.extend(self._encode_batch(encoder, items))
            encoded_since_report += len(items)
            now = time.monotonic()
            if now - last_encode_report >= 5.0:
                elapsed = max(now - encode_started, 1e-6)
                speed = encoded_since_report / elapsed
                target = self.block_size
                remaining = max(target - len(encoded_buffer), 0)
                eta = remaining / speed if speed > 0 else float("inf")
                print(
                    f"[encode] {len(encoded_buffer):,}/{target:,} latent "
                    f"| {speed:.2f} image/s | ETA {eta:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                last_encode_report = now

        def flush_all() -> None:
            for bucket_key in list(pending_by_bucket):
                flush_bucket(bucket_key)

        def pop_block(size: int) -> list[EncodedImage]:
            encoded_buffer.sort(
                key=lambda item: (
                    item.source_shard_index,
                    item.source_sample_index,
                )
            )
            block = encoded_buffer[:size]
            del encoded_buffer[:size]
            return block

        try:
            shard_total = len(sources)
            for shard_offset, shard_path in enumerate(prefetcher):
                shard_index = start_shard + shard_offset
                source = sources[shard_offset]
                label = Path(str(source).replace("\\", "/")).name
                print(
                    f"[shard] {shard_offset + 1:,}/{shard_total:,}: {label}",
                    file=sys.stderr,
                    flush=True,
                )
                external_metadata = None
                if self.metadata_index is not None:
                    metadata_started = time.monotonic()
                    print(
                        f"[data] {label}: loading metadata partition...",
                        file=sys.stderr,
                        flush=True,
                    )
                    external_metadata = self.metadata_index.load_for_source(source)
                    print(
                        f"[data] {label}: loaded "
                        f"{len(external_metadata):,} metadata rows in "
                        f"{time.monotonic() - metadata_started:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                raw_progress: dict[str, int] = {}
                raw_samples = _threaded_prefetch(
                    iter_raw_tar(
                        shard_path,
                        require_metadata=self.require_metadata,
                        external_metadata=external_metadata,
                        progress_label=label,
                        progress_state=raw_progress,
                    ),
                    capacity=self.decode_prefetch,
                )
                shard_started = time.monotonic()
                accepted = 0
                skipped_rating = 0
                skipped_upscale = 0
                skipped_invalid = 0
                last_data_report = shard_started
                for sample_index, (key, image, metadata) in enumerate(raw_samples):
                    if (
                        self.resume_epoch == epoch
                        and shard_index == self.resume_shard_index
                        and sample_index <= self.resume_sample_index
                    ):
                        continue
                    if self.allowed_ratings is not None:
                        rating = _normalized_rating(metadata.get("rating"))
                        if rating not in self.allowed_ratings:
                            skipped_rating += 1
                            continue
                    try:
                        bucket = choose_bucket(
                            image.width, image.height, self.buckets
                        )
                        required_scale = max(
                            bucket.width / image.width,
                            bucket.height / image.height,
                        )
                        if required_scale > self.max_upscale:
                            skipped_upscale += 1
                            continue
                        original_size = (image.width, image.height)
                        pixels, crop_box = prepare_image(image, bucket)
                    except (OSError, ValueError):
                        skipped_invalid += 1
                        continue
                    accepted += 1

                    record = dict(metadata)
                    record.update(
                        {
                            "bucket": bucket.key,
                            "original_size": list(original_size),
                            "crop_box": list(crop_box),
                            "source_shard": sources[shard_offset],
                            "source_member": key,
                            "vae": "Wan2.2_TI2V_5B_f16c48",
                        }
                    )
                    item = PreparedImage(
                        source_shard_index=shard_index,
                        source_sample_index=sample_index,
                        sample_id=f"{shard_index:06d}-{sample_index:08d}",
                        pixels=pixels,
                        metadata=record,
                        bucket=bucket.key,
                    )
                    bucket_items = pending_by_bucket.setdefault(bucket.key, [])
                    bucket_items.append(item)
                    if len(bucket_items) >= self._effective_encode_batch_size:
                        flush_bucket(bucket.key)

                    now = time.monotonic()
                    if now - last_data_report >= 5.0:
                        scanned = raw_progress.get("scanned", sample_index + 1)
                        total = raw_progress.get("total", 0)
                        elapsed = max(now - shard_started, 1e-6)
                        rate = scanned / elapsed
                        eta = (
                            max(total - scanned, 0) / rate
                            if total and rate > 0
                            else 0.0
                        )
                        reserved = (
                            torch.cuda.memory_reserved() / 1024**3
                            if torch.cuda.is_available()
                            else 0.0
                        )
                        print(
                            f"[data] {label}: {scanned:,}/{total:,} scanned "
                            f"| {accepted:,} accepted "
                            f"| skip rating={skipped_rating:,}, "
                            f"upscale={skipped_upscale:,}, "
                            f"invalid={skipped_invalid:,} "
                            f"| {rate:.1f} image/s | ETA {eta:.0f}s "
                            f"| CUDA reserved={reserved:.1f} GiB",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_data_report = now

                    pending_count = sum(
                        len(values) for values in pending_by_bucket.values()
                    )
                    if len(encoded_buffer) + pending_count < self.block_size:
                        continue
                    flush_all()
                    while len(encoded_buffer) >= self.block_size:
                        block = pop_block(self.block_size)
                        print(
                            f"[encode] block ready: {len(block):,} latents; "
                            "offloading Wan VAE and starting DiT consumption",
                            file=sys.stderr,
                            flush=True,
                        )
                        self._offload_encoder(encoder)
                        for encoded in block:
                            yield {
                                "latent": encoded.latent,
                                "caption": self._caption(encoded),
                                "bucket": encoded.bucket,
                                "index": global_index,
                                "sample_id": encoded.sample_id,
                                "stream_epoch": epoch,
                                "source_shard_index": encoded.source_shard_index,
                                "source_sample_index": encoded.source_sample_index,
                            }
                            global_index += 1
                        encoder = self._move_encoder_to_compute()
                        encode_started = time.monotonic()
                        encoded_since_report = 0
                        last_encode_report = encode_started

                elapsed = max(time.monotonic() - shard_started, 1e-6)
                total = raw_progress.get("total", 0)
                print(
                    f"[data] {label}: complete {total:,}/{total:,} scanned, "
                    f"{accepted:,} accepted in {elapsed:.1f}s "
                    f"({total / elapsed:.1f} image/s)",
                    file=sys.stderr,
                    flush=True,
                )

            flush_all()
            usable = (
                len(encoded_buffer) // self.accumulation_multiple
            ) * self.accumulation_multiple
            if usable:
                block = pop_block(usable)
                self._offload_encoder(encoder)
                for encoded in block:
                    yield {
                        "latent": encoded.latent,
                        "caption": self._caption(encoded),
                        "bucket": encoded.bucket,
                        "index": global_index,
                        "sample_id": encoded.sample_id,
                        "stream_epoch": epoch,
                        "source_shard_index": encoded.source_shard_index,
                        "source_sample_index": encoded.source_sample_index,
                    }
                    global_index += 1
        finally:
            self._offload_encoder(encoder)
            self.resume_epoch = None
