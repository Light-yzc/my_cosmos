import io
import json
import tarfile

import torch
from PIL import Image

from my_sd.data.captions import DanbooruCaptionConfig, DanbooruCaptioner
from my_sd.data.raw_stream import RollingWanDataset, iter_raw_tar


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), color).save(buffer, format="WEBP")
    return buffer.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _make_raw_tar(path, *, count: int = 3, include_missing: bool = False) -> None:
    with tarfile.open(path, "w") as archive:
        for index in range(count):
            key = f"{1000 + index}"
            _add_bytes(
                archive,
                f"{key}.webp",
                _image_bytes((index * 20, 64, 128)),
            )
            _add_bytes(
                archive,
                f"{key}.json",
                json.dumps(
                    {
                        "rating": "g",
                        "general_tags": ["1girl", f"sample_{index}"],
                    }
                ).encode(),
            )
        if include_missing:
            _add_bytes(archive, "missing.webp", _image_bytes((255, 0, 0)))


class FakeEncoder:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.device = "cpu"

    def move_to(self, device: str, dtype: str | None = None) -> None:
        self.device = device
        self.events.append(("move", device))

    def offload_to_cpu(self) -> None:
        self.device = "cpu"
        self.events.append(("offload", "cpu"))

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        self.events.append(("encode", len(images)))
        height, width = images.shape[-2:]
        return torch.zeros(
            len(images),
            48,
            height // 16,
            width // 16,
            dtype=torch.float32,
        )


def test_raw_tar_requires_caption_metadata(tmp_path) -> None:
    path = tmp_path / "raw.tar"
    _make_raw_tar(path, count=1, include_missing=True)
    required = list(iter_raw_tar(path, require_metadata=True))
    optional = list(iter_raw_tar(path, require_metadata=False))
    assert [sample[0] for sample in required] == ["1000"]
    assert {sample[0] for sample in optional} == {"1000", "missing"}


def test_rolling_dataset_offloads_before_yield_and_drops_partial_accumulation(
    tmp_path,
) -> None:
    path = tmp_path / "raw.tar"
    _make_raw_tar(path, count=3)
    encoder = FakeEncoder()
    dataset = RollingWanDataset(
        [str(path)],
        encoder_factory=lambda: encoder,
        cache_dir=tmp_path / "cache",
        captioner=DanbooruCaptioner(
            DanbooruCaptionConfig(
                general_tag_dropout=0.0,
                character_tag_dropout=0.0,
                shuffle_general_tags=False,
            )
        ),
        resolution_stage="512",
        block_size=2,
        encode_batch_size=1,
        accumulation_multiple=2,
        max_upscale=1.0,
        shuffle_shards=False,
        delete_after_use=False,
    )
    iterator = iter(dataset)
    first = next(iterator)
    assert encoder.device == "cpu"
    assert ("offload", "cpu") in encoder.events
    remaining = list(iterator)
    samples = [first, *remaining]
    assert len(samples) == 2
    assert [sample["source_sample_index"] for sample in samples] == [0, 1]
    assert samples[0]["latent"].shape == (48, 32, 32)
    assert samples[0]["caption"] == "rating:g, 1girl, sample 0"


def test_rolling_resume_skips_trained_source_samples(tmp_path) -> None:
    path = tmp_path / "raw.tar"
    _make_raw_tar(path, count=3)
    encoder = FakeEncoder()
    dataset = RollingWanDataset(
        [str(path)],
        encoder_factory=lambda: encoder,
        cache_dir=tmp_path / "cache",
        resolution_stage="512",
        block_size=2,
        encode_batch_size=2,
        accumulation_multiple=2,
        max_upscale=1.0,
        shuffle_shards=False,
        delete_after_use=False,
    )
    dataset.set_resume_cursor(epoch=0, shard_index=0, sample_index=0)
    samples = list(dataset)
    assert [sample["source_sample_index"] for sample in samples] == [1, 2]
