from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

_BUCKET_NAME = re.compile(r"^\d{4}$")


def deepghs_bucket_from_source(source: str | Path) -> str:
    """Return the four-digit DeepGHS image bucket from a shard path or URL."""
    value = str(source)
    path = urlparse(value).path if "://" in value else value
    bucket = PurePosixPath(path.replace("\\", "/")).stem
    if not _BUCKET_NAME.fullmatch(bucket):
        raise ValueError(
            f"DeepGHS shard must end in images/NNNN.tar, got {value!r}"
        )
    return bucket


class DeepGHSMetadataIndex:
    """Loads one compact metadata partition for each DeepGHS image tar."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def files_for_bucket(self, bucket: str) -> list[Path]:
        direct = self.root / f"{bucket}.parquet"
        if direct.is_file():
            return [direct]
        partition = self.root / f"bucket={bucket}"
        return sorted(partition.glob("*.parquet"))

    def load_for_source(self, source: str | Path) -> dict[str, dict[str, object]]:
        bucket = deepghs_bucket_from_source(source)
        files = self.files_for_bucket(bucket)
        if not files:
            raise FileNotFoundError(
                f"metadata partition for bucket {bucket} is missing under "
                f"{self.root}; run scripts/prepare_deepghs_metadata.py first"
            )
        try:
            import duckdb
        except ImportError as error:
            raise RuntimeError(
                "DeepGHS metadata requires duckdb; run `uv sync --extra train`"
            ) from error

        columns = (
            "id, rating, score, image_width, image_height, "
            "tag_string_general, tag_string_character, "
            "tag_string_copyright, tag_string_artist, tag_string_meta"
        )
        placeholders = ", ".join("?" for _ in files)
        query = (
            f"SELECT {columns} FROM read_parquet([{placeholders}], "
            "union_by_name=true)"
        )
        rows = duckdb.connect().execute(
            query, [str(path) for path in files]
        ).fetchall()
        names = [name.strip() for name in columns.split(",")]
        return {
            str(row[0]): dict(zip(names, row, strict=True))
            for row in rows
        }
