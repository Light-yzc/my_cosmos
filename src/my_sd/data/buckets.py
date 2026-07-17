from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ResolutionBucket:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bucket dimensions must be positive")
        if self.width % 32 or self.height % 32:
            raise ValueError(
                "Bucket dimensions must be multiples of 32 "
                "(Wan f16 followed by DiT patch-size 2)"
            )

    @property
    def key(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def latent_size(self) -> tuple[int, int]:
        return self.width // 16, self.height // 16

    @property
    def token_count(self) -> int:
        return (self.width // 32) * (self.height // 32)


LOW_RES_BUCKETS: tuple[ResolutionBucket, ...] = (
    ResolutionBucket(512, 512),
    ResolutionBucket(448, 576),
    ResolutionBucket(576, 448),
    ResolutionBucket(416, 608),
    ResolutionBucket(608, 416),
    ResolutionBucket(384, 672),
    ResolutionBucket(672, 384),
)

DEFAULT_BUCKETS: tuple[ResolutionBucket, ...] = (
    ResolutionBucket(768, 768),
    ResolutionBucket(672, 896),
    ResolutionBucket(896, 672),
    ResolutionBucket(640, 960),
    ResolutionBucket(960, 640),
    ResolutionBucket(576, 1024),
    ResolutionBucket(1024, 576),
)

HIGH_RES_BUCKETS: tuple[ResolutionBucket, ...] = (
    ResolutionBucket(1024, 1024),
    ResolutionBucket(864, 1152),
    ResolutionBucket(1152, 864),
    ResolutionBucket(832, 1248),
    ResolutionBucket(1248, 832),
    ResolutionBucket(736, 1312),
    ResolutionBucket(1312, 736),
)


def parse_bucket(value: str) -> ResolutionBucket:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        return ResolutionBucket(int(width_text), int(height_text))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid bucket {value!r}; expected WIDTHxHEIGHT") from error


def choose_bucket(
    image_width: int,
    image_height: int,
    buckets: Sequence[ResolutionBucket] = DEFAULT_BUCKETS,
) -> ResolutionBucket:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not buckets:
        raise ValueError("At least one resolution bucket is required")
    image_ratio = image_width / image_height
    return min(
        buckets,
        key=lambda bucket: (
            abs(math.log(image_ratio / bucket.aspect_ratio)),
            abs(image_width * image_height - bucket.width * bucket.height),
        ),
    )


def cover_resize_and_center_crop(
    image_width: int,
    image_height: int,
    bucket: ResolutionBucket,
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Returns PIL resize dimensions and a centered crop box."""
    scale = max(bucket.width / image_width, bucket.height / image_height)
    resized_width = max(bucket.width, math.ceil(image_width * scale))
    resized_height = max(bucket.height, math.ceil(image_height * scale))
    left = (resized_width - bucket.width) // 2
    top = (resized_height - bucket.height) // 2
    crop = (left, top, left + bucket.width, top + bucket.height)
    return (resized_width, resized_height), crop


def bucket_map(
    buckets: Iterable[ResolutionBucket] = DEFAULT_BUCKETS,
) -> dict[str, ResolutionBucket]:
    return {bucket.key: bucket for bucket in buckets}
