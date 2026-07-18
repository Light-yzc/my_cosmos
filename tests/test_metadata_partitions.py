import duckdb

from scripts.prepare_deepghs_metadata import (
    _compact_partitions,
    _partition_stats,
)
from my_sd.training.preflight import _metadata_partition_stats


def test_multiple_parquet_files_count_as_one_logical_bucket(tmp_path) -> None:
    for bucket in range(5):
        directory = tmp_path / f"bucket={bucket:04d}"
        directory.mkdir()
        for part in range(3):
            (directory / f"data_{part}.parquet").write_bytes(b"stub")
    assert _partition_stats(tmp_path) == (5, 15)
    assert _metadata_partition_stats(tmp_path) == (5, 15)


def test_fragmented_buckets_are_compacted_before_drive_copy(tmp_path) -> None:
    build = tmp_path / "build"
    connection = duckdb.connect()
    for bucket in range(2):
        directory = build / f"bucket={bucket:04d}"
        directory.mkdir(parents=True)
        for part in range(3):
            destination = str(directory / f"data_{part}.parquet").replace(
                "\\", "/"
            )
            connection.execute(
                f"COPY (SELECT {part} AS id, 'tag' AS tag_string) "
                f"TO '{destination}' (FORMAT PARQUET)"
            )
    buckets, files = _compact_partitions(
        connection,
        build,
        minimum_buckets=2,
        maximum_buckets=2,
    )
    assert buckets == 2
    assert files <= 4
