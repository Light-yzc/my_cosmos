from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

INDEX_VERSION = 2
DEFAULT_REPO = "deepghs/danbooru2024-webp-4Mpixel"
DEFAULT_FILENAME = "metadata.parquet"


def _partition_stats(root: Path) -> tuple[int, int]:
    """Return logical bucket count and physical parquet file count."""
    files = list(root.glob("bucket=*/*.parquet"))
    direct = list(root.glob("????.parquet"))
    buckets = {
        path.parent.name.removeprefix("bucket=")
        for path in files
        if path.parent.name.startswith("bucket=")
    }
    buckets.update(path.stem for path in direct)
    return len(buckets), len(files) + len(direct)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _compact_partitions(
    connection,
    build_dir: Path,
    *,
    minimum_buckets: int = 900,
    maximum_buckets: int = 1100,
) -> tuple[int, int]:
    """Rewrite fragmented partition output to approximately one file per bucket."""
    compact = build_dir.with_name(build_dir.name + ".compacted")
    if compact.exists():
        shutil.rmtree(compact)
    source_glob = _sql_path(build_dir / "bucket=*" / "*.parquet")
    compact_sql = _sql_path(compact)
    connection.execute("SET partitioned_write_max_open_files = 2000")
    connection.execute(
        f"""
        COPY (
          SELECT *
          FROM read_parquet(
            '{source_glob}',
            hive_partitioning = true,
            union_by_name = true
          )
        ) TO '{compact_sql}' (
          FORMAT PARQUET,
          PARTITION_BY (bucket),
          COMPRESSION ZSTD
        )
        """
    )
    bucket_count, file_count = _partition_stats(compact)
    # DuckDB can retain Windows handles for freshly written parquet files until
    # the connection closes, which prevents cleanup or the atomic rename.
    connection.close()
    if not minimum_buckets <= bucket_count <= maximum_buckets:
        shutil.rmtree(compact)
        raise RuntimeError(
            f"metadata compaction produced {bucket_count} buckets and "
            f"{file_count} files"
        )
    shutil.rmtree(build_dir)
    compact.replace(build_dir)
    return bucket_count, file_count


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _expression(
    columns: set[str],
    candidates: tuple[str, ...],
    *,
    default: str,
    alias: str,
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return f"{_quoted(candidate)} AS {_quoted(alias)}"
    return f"{default} AS {_quoted(alias)}"


def _existing_index_is_current(output: Path, repo: str, filename: str) -> bool:
    manifest = output / "_index_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    bucket_count, file_count = _partition_stats(output)
    return (
        value.get("version") == INDEX_VERSION
        and value.get("source_repo") == repo
        and value.get("source_filename") == filename
        and 900 <= bucket_count <= 1100
        and file_count >= bucket_count
    )


def main() -> None:
    import duckdb
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import enable_progress_bars

    parser = argparse.ArgumentParser(
        description=(
            "Download the metadata shipped with DeepGHS Danbooru 2024 and "
            "partition it to match its 1000 image tar buckets."
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
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if (
        not args.force
        and _existing_index_is_current(
            args.output, args.repo, args.filename
        )
    ):
        print(f"metadata index already exists and is current: {args.output}")
        return
    if args.output.exists():
        print(
            "legacy, fragmented, or mismatched metadata index detected; "
            "building a same-source DeepGHS v2 index before replacing it",
            flush=True,
        )
    build_bucket_count, build_file_count = _partition_stats(args.build_dir)
    reuse_build = (
        not args.force
        and 900 <= build_bucket_count <= 1100
        and build_file_count >= build_bucket_count
    )
    if args.build_dir.exists() and not reuse_build:
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    if reuse_build:
        print(
            f"reusing completed build: {build_bucket_count} buckets across "
            f"{build_file_count} parquet files",
            flush=True,
        )
    args.download_dir.mkdir(parents=True, exist_ok=True)

    enable_progress_bars()
    print(
        f"downloading {args.repo}/{args.filename} with progress...",
        flush=True,
    )
    parquet = Path(
        hf_hub_download(
            repo_id=args.repo,
            filename=args.filename,
            repo_type="dataset",
            local_dir=args.download_dir,
            token=os.environ.get("HF_TOKEN"),
        )
    ).resolve()
    connection = duckdb.connect()
    description = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet)]
    ).fetchall()
    columns = {str(row[0]) for row in description}
    if "id" not in columns:
        raise RuntimeError(
            f"{parquet} has no `id` column; found {sorted(columns)}"
        )

    select_values = [
        'printf(\'%04d\', "id" % 1000) AS "bucket"',
        '"id" AS "id"',
        _expression(columns, ("rating",), default="'u'", alias="rating"),
        _expression(columns, ("score",), default="0", alias="score"),
        _expression(
            columns,
            ("image_width", "width"),
            default="NULL",
            alias="image_width",
        ),
        _expression(
            columns,
            ("image_height", "height"),
            default="NULL",
            alias="image_height",
        ),
        _expression(
            columns,
            ("tag_string_general", "tag_string"),
            default="''",
            alias="tag_string_general",
        ),
        _expression(
            columns,
            ("tag_string_character",),
            default="''",
            alias="tag_string_character",
        ),
        _expression(
            columns,
            ("tag_string_copyright",),
            default="''",
            alias="tag_string_copyright",
        ),
        _expression(
            columns,
            ("tag_string_artist",),
            default="''",
            alias="tag_string_artist",
        ),
        _expression(
            columns,
            ("tag_string_meta",),
            default="''",
            alias="tag_string_meta",
        ),
    ]
    predicates: list[str] = []
    for name in ("is_deleted", "is_banned", "is_flagged"):
        if name in columns:
            predicates.append(f"NOT coalesce({_quoted(name)}, false)")
    tag_source = (
        "tag_string_general"
        if "tag_string_general" in columns
        else "tag_string"
    )
    if tag_source in columns:
        predicates.append(f"coalesce({_quoted(tag_source)}, '') <> ''")
    where = " AND ".join(predicates) if predicates else "true"
    parquet_sql = _sql_path(parquet)
    output_sql = _sql_path(args.build_dir)
    query = f"""
    COPY (
      SELECT {", ".join(select_values)}
      FROM read_parquet('{parquet_sql}')
      WHERE {where}
    ) TO '{output_sql}' (
      FORMAT PARQUET,
      PARTITION_BY (bucket),
      COMPRESSION ZSTD
    )
    """
    if reuse_build:
        elapsed = 0.0
    else:
        print(
            "building 1000 same-source metadata buckets "
            "(one full parquet scan)...",
            flush=True,
        )
        connection.execute("SET threads = 1")
        # Keeping all 1000 logical partitions open avoids DuckDB repeatedly
        # closing/reopening buckets and emitting thousands of tiny files.
        connection.execute("SET partitioned_write_max_open_files = 2000")
        connection.execute("SET enable_progress_bar = true")
        connection.execute("SET progress_bar_time = 1000")
        started = time.monotonic()
        connection.execute(query)
        elapsed = time.monotonic() - started
    bucket_count, file_count = _partition_stats(args.build_dir)
    if not 900 <= bucket_count <= 1100:
        raise RuntimeError(
            f"expected roughly 1000 metadata buckets, got {bucket_count} "
            f"across {file_count} parquet files"
        )
    if file_count > bucket_count * 2:
        print(
            f"compacting {file_count} parquet fragments into roughly "
            f"{bucket_count} files before the Drive copy...",
            flush=True,
        )
        compact_started = time.monotonic()
        bucket_count, file_count = _compact_partitions(
            connection,
            args.build_dir,
        )
        print(
            f"metadata compaction completed in "
            f"{time.monotonic() - compact_started:.1f}s: "
            f"{bucket_count} buckets, {file_count} files",
            flush=True,
        )
    manifest = {
        "version": INDEX_VERSION,
        "source_repo": args.repo,
        "source_filename": args.filename,
        "partition_count": bucket_count,
        "parquet_file_count": file_count,
        "columns": sorted(columns),
    }
    (args.build_dir / "_index_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if not reuse_build:
        print(f"metadata partition build completed in {elapsed:.1f}s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    replacement = args.output.with_name(args.output.name + ".replacement")
    if replacement.exists():
        shutil.rmtree(replacement)
    shutil.copytree(args.build_dir, replacement)
    if args.output.exists():
        shutil.rmtree(args.output)
    replacement.replace(args.output)
    print(
        f"wrote {bucket_count} metadata buckets "
        f"({file_count} parquet files) to {args.output}",
        flush=True,
    )
    if not args.keep_source:
        shutil.rmtree(args.download_dir)
    shutil.rmtree(args.build_dir)


if __name__ == "__main__":
    main()
