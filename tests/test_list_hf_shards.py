from scripts.list_hf_shards import list_repo_tar_files


class FlakyApi:
    def __init__(self) -> None:
        self.calls = 0

    def list_repo_files(self, *args, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise TimeoutError("temporary timeout")
        return [
            "README.md",
            "train/00002.tar",
            "train/00001.tar",
            "val/00001.tar",
        ]


def test_shard_listing_retries_and_filters_split(monkeypatch) -> None:
    api = FlakyApi()
    monkeypatch.setattr("scripts.list_hf_shards.time.sleep", lambda _: None)
    files = list_repo_tar_files(
        api,
        repo="owner/dataset",
        split="train",
        revision="main",
        retries=4,
    )
    assert api.calls == 3
    assert files == ["train/00001.tar", "train/00002.tar"]
