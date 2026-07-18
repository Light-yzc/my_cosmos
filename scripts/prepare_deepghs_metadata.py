from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    import duckdb
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import enable_progress_bars

    parser = argparse.ArgumentParser(
        description=(
            "Download Danbooru 2024 metadata and partition it to match the "
            "1000 DeepGHS image tar buckets."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/content/deepghs_metadata_source"),
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/content/deepghs_metadata_build"),
    )
    parser.add_argument("--repo", default="p1atdev/danbooru-2024")
    parser.add_argument("--keep-source", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.rglob("*.parquet")):
        print(f"metadata index already exists: {args.output}")
        return
    if args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    enable_progress_bars()
    print("downloading 35 metadata parquet shards...", flush=True)
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=args.download_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    parquet_glob = str(
        (args.download_dir / "data" / "*.parquet").resolve()
    ).replace("\\", "/").replace("'", "''")
    output = str(args.build_dir.resolve()).replace("\\", "/").replace("'", "''")
    query = f"""
    COPY (
      SELECT
        printf('%04d', id % 1000) AS bucket,
        id, rating, score, image_width, image_height,
        tag_string_general, tag_string_character,
        tag_string_copyright, tag_string_artist, tag_string_meta
      FROM read_parquet('{parquet_glob}', union_by_name=true)
      WHERE NOT coalesce(is_deleted, false)
        AND NOT coalesce(is_banned, false)
        AND NOT coalesce(is_flagged, false)
        AND coalesce(tag_string_general, '') <> ''
    ) TO '{output}' (
      FORMAT PARQUET,
      PARTITION_BY (bucket),
      COMPRESSION ZSTD
    )
    """
    print(
        "building 1000 metadata partitions (one full metadata scan)...",
        flush=True,
    )
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar = true")
    connection.execute("SET progress_bar_time = 1000")
    started = __import__("time").monotonic()
    connection.execute(query)
    elapsed = __import__("time").monotonic() - started
    print(f"metadata partition build completed in {elapsed:.1f}s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        shutil.rmtree(args.output)
    shutil.copytree(args.build_dir, args.output)
    count = len(list(args.output.rglob("*.parquet")))
    print(f"wrote {count} metadata partitions to {args.output}")
    if not args.keep_source:
        shutil.rmtree(args.download_dir)
    shutil.rmtree(args.build_dir)


if __name__ == "__main__":
    main()
