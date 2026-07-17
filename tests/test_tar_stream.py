import torch

from my_sd.data.captions import DanbooruCaptionConfig, DanbooruCaptioner
from my_sd.data.latent_dataset import collate_latents
from my_sd.data.tar_stream import (
    LatentTarWriter,
    StreamingLatentDataset,
    iter_latent_tar,
)


def test_latent_tar_round_trip_and_stream_cursor(tmp_path) -> None:
    path = tmp_path / "latent.tar"
    latent = torch.randn(48, 8, 12)
    with LatentTarWriter(path) as writer:
        writer.add(
            "sample-a",
            latent,
            {
                "bucket": "192x128",
                "general_tags": ["1girl", "blue_hair"],
            },
        )

    decoded = list(iter_latent_tar(path))
    assert decoded[0]["sample_id"] == "sample-a"
    assert decoded[0]["latent"].shape == latent.shape
    assert decoded[0]["latent"].dtype == torch.float16

    dataset = StreamingLatentDataset(
        [str(path)],
        cache_dir=tmp_path / "cache",
        captioner=DanbooruCaptioner(
            DanbooruCaptionConfig(
                general_tag_dropout=0.0,
                character_tag_dropout=0.0,
                shuffle_general_tags=False,
            )
        ),
        sample_shuffle_buffer=1,
        delete_after_use=False,
        shuffle_shards=False,
    )
    sample = next(iter(dataset))
    assert sample["caption"] == "1girl, blue hair"
    assert sample["source_shard_index"] == 0
    batch = collate_latents([sample])
    assert batch["source_sample_index"] == [0]

    dataset.set_resume_cursor(epoch=0, shard_index=0, sample_index=0)
    assert list(dataset) == []

