from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from safetensors.torch import load_file
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .captions import DanbooruCaptioner


def _resolved_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


class LatentManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        captioner: DanbooruCaptioner | None = None,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_dir = self.manifest_path.parent
        self.captioner = captioner or DanbooruCaptioner()
        self.horizontal_flip_probability = horizontal_flip_probability
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if horizontal_flip_probability:
            raise ValueError(
                "Do not flip an already encoded Wan latent. Pre-encode flipped RGB "
                "variants as separate manifest records instead."
            )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"No records found in {self.manifest_path}")
        for index, record in enumerate(self.records):
            if "latent_path" not in record or "bucket" not in record:
                raise ValueError(
                    f"Manifest record {index} needs latent_path and bucket fields"
                )

    def __len__(self) -> int:
        return len(self.records)

    def bucket_for_index(self, index: int) -> str:
        return str(self.records[index]["bucket"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        latent_path = _resolved_path(str(record["latent_path"]), self.manifest_dir)
        tensors = load_file(str(latent_path), device="cpu")
        if "latent" not in tensors:
            raise KeyError(f"{latent_path} does not contain a 'latent' tensor")
        latent = tensors["latent"]
        if latent.ndim == 4 and latent.shape[1] == 1:
            latent = latent.squeeze(1)
        if latent.ndim != 3:
            raise ValueError(
                f"Expected cached latent [C,H,W], got {tuple(latent.shape)} in {latent_path}"
            )
        if any(
            key in record
            for key in (
                "general_tags",
                "tag_string_general",
                "character_tags",
                "tag_string_character",
            )
        ):
            caption = self.captioner.compose(record)
        else:
            caption = str(record.get("caption", ""))
        return {
            "latent": latent,
            "caption": caption,
            "bucket": str(record["bucket"]),
            "index": index,
        }


class BucketBatchSampler(Sampler[list[int]]):
    """Groups variable-aspect samples so tensors in a batch remain stackable."""

    def __init__(
        self,
        dataset: LatentManifestDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups: dict[str, list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            groups[self.dataset.bucket_for_index(index)].append(index)

        batches: list[list[int]] = []
        for indices in groups.values():
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        counts: dict[str, int] = defaultdict(int)
        for index in range(len(self.dataset)):
            counts[self.dataset.bucket_for_index(index)] += 1
        if self.drop_last:
            return sum(count // self.batch_size for count in counts.values())
        return sum((count + self.batch_size - 1) // self.batch_size for count in counts.values())


def collate_latents(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    buckets = {sample["bucket"] for sample in samples}
    if len(buckets) != 1:
        raise ValueError(f"Mixed buckets in one batch: {sorted(buckets)}")
    batch = {
        "latents": torch.stack([sample["latent"] for sample in samples]),
        "captions": [sample["caption"] for sample in samples],
        "bucket": samples[0]["bucket"],
        "indices": torch.tensor([sample["index"] for sample in samples]),
    }
    optional_keys = (
        "sample_id",
        "stream_epoch",
        "source_shard_index",
        "source_sample_index",
        "resume_shard_index",
        "resume_sample_index",
    )
    for key in optional_keys:
        if all(key in sample for sample in samples):
            batch[key] = [sample[key] for sample in samples]
    return batch
