from my_sd.data.buckets import (
    DEFAULT_BUCKETS,
    HIGH_RES_BUCKETS,
    LOW_RES_BUCKETS,
    choose_bucket,
    cover_resize_and_center_crop,
)


def test_default_buckets_match_one_gpu_pixel_budget() -> None:
    assert {bucket.key for bucket in DEFAULT_BUCKETS} == {
        "768x768",
        "672x896",
        "896x672",
        "640x960",
        "960x640",
        "576x1024",
        "1024x576",
    }
    assert all(576 <= bucket.token_count <= 600 for bucket in DEFAULT_BUCKETS)
    assert all(247 <= bucket.token_count <= 256 for bucket in LOW_RES_BUCKETS)
    assert all(bucket.token_count <= 1024 for bucket in HIGH_RES_BUCKETS)


def test_choose_portrait_and_cover_crop() -> None:
    bucket = choose_bucket(1200, 1800)
    assert bucket.key == "640x960"
    resize, crop = cover_resize_and_center_crop(1200, 1800, bucket)
    assert resize == (640, 960)
    assert crop == (0, 0, 640, 960)


def test_choose_wide_bucket() -> None:
    assert choose_bucket(1920, 1080).key == "1024x576"
