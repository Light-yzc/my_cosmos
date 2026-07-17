from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class RandomLike(Protocol):
    def random(self) -> float: ...
    def shuffle(self, x: list[Any]) -> None: ...


def _as_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        separator = "," if "," in value else None
        return [part.strip() for part in value.split(separator) if part.strip()]
    if isinstance(value, Sequence):
        return [str(part).strip() for part in value if str(part).strip()]
    raise TypeError(f"Tag field must be a string or sequence, got {type(value).__name__}")


@dataclass(slots=True)
class DanbooruCaptionConfig:
    general_tag_dropout: float = 0.10
    character_tag_dropout: float = 0.02
    shuffle_general_tags: bool = True
    max_tags: int = 128
    replace_underscores: bool = True
    include_artist: bool = True
    include_rating: bool = True
    include_quality: bool = True

    def validate(self) -> None:
        for name in ("general_tag_dropout", "character_tag_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.max_tags < 1:
            raise ValueError("max_tags must be positive")


class DanbooruCaptioner:
    """Builds a fresh tag ordering on every sample access."""

    def __init__(self, config: DanbooruCaptionConfig | None = None) -> None:
        self.config = config or DanbooruCaptionConfig()
        self.config.validate()

    @staticmethod
    def _drop(
        tags: list[str],
        probability: float,
        rng: RandomLike,
    ) -> list[str]:
        return [tag for tag in tags if rng.random() >= probability]

    def _format(self, tag: str) -> str:
        return tag.replace("_", " ") if self.config.replace_underscores else tag

    def compose(
        self,
        record: Mapping[str, object],
        rng: RandomLike = random,
    ) -> str:
        general = _as_tags(
            record.get("general_tags", record.get("tag_string_general"))
        )
        characters = _as_tags(
            record.get("character_tags", record.get("tag_string_character"))
        )
        copyrights = _as_tags(
            record.get("copyright_tags", record.get("tag_string_copyright"))
        )
        artists = _as_tags(record.get("artist_tags", record.get("tag_string_artist")))
        meta = _as_tags(record.get("meta_tags", record.get("tag_string_meta")))

        general = self._drop(general, self.config.general_tag_dropout, rng)
        characters = self._drop(
            characters, self.config.character_tag_dropout, rng
        )
        if self.config.shuffle_general_tags:
            rng.shuffle(general)

        prefix: list[str] = []
        quality = str(record.get("quality", "")).strip()
        rating = str(record.get("rating", "")).strip()
        if self.config.include_quality and quality:
            prefix.append(quality if "quality" in quality else f"{quality} quality")
        if self.config.include_rating and rating:
            prefix.append(rating if rating.startswith("rating:") else f"rating:{rating}")
        if self.config.include_artist:
            prefix.extend(f"artist:{tag}" for tag in artists)

        ordered = prefix + characters + copyrights + general + meta
        deduplicated = list(dict.fromkeys(tag for tag in ordered if tag))
        return ", ".join(self._format(tag) for tag in deduplicated[: self.config.max_tags])

