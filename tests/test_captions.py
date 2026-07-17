import random

from my_sd.data.captions import DanbooruCaptionConfig, DanbooruCaptioner


def test_caption_preserves_anchors_and_formats_tags() -> None:
    captioner = DanbooruCaptioner(
        DanbooruCaptionConfig(
            general_tag_dropout=0.0,
            character_tag_dropout=0.0,
            shuffle_general_tags=False,
        )
    )
    caption = captioner.compose(
        {
            "rating": "safe",
            "quality": "high",
            "artist_tags": ["some_artist"],
            "character_tags": "hatsune_miku",
            "copyright_tags": "vocaloid",
            "general_tags": "1girl blue_hair looking_at_viewer",
        },
        rng=random.Random(1),
    )
    assert caption == (
        "high quality, rating:safe, artist:some artist, hatsune miku, "
        "vocaloid, 1girl, blue hair, looking at viewer"
    )


def test_caption_shuffle_is_seed_reproducible() -> None:
    captioner = DanbooruCaptioner(
        DanbooruCaptionConfig(
            general_tag_dropout=0.0,
            character_tag_dropout=0.0,
            shuffle_general_tags=True,
        )
    )
    record = {"general_tags": ["a", "b", "c", "d"]}
    assert captioner.compose(record, random.Random(42)) == captioner.compose(
        record, random.Random(42)
    )

