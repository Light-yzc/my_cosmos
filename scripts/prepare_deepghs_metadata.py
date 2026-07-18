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
    partitions = list(output.glob("bucket=*/*.parquet"))
    partitions.extend(output.glob("????.parquet"))
    return (
        value.get("version") == INDEX_VERSION
        and value.get("source_repo") == repo
        and value.get("source_filename") == filename
        and 900 <= len(partitions) <= 1100
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
    if args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True)
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
    parquet_sql = str(parquet).replace("\\", "/").replace("'", "''")
    output_sql = (
        str(args.build_dir.resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )
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
    print(
        "building 1000 same-source metadata partitions "
        "(one full parquet scan)...",
        flush=True,
    )
    connection.execute("SET threads = 1")
    connection.execute("SET enable_progress_bar = true")
    connection.execute("SET progress_bar_time = 1000")
    started = time.monotonic()
    connection.execute(query)
    elapsed = time.monotonic() - started
    partitions = list(args.build_dir.glob("bucket=*/*.parquet"))
    if not 900 <= len(partitions) <= 1100:
        raise RuntimeError(
            f"expected roughly 1000 metadata partitions, got {len(partitions)}"
        )
    manifest = {
        "version": INDEX_VERSION,
        "source_repo": args.repo,
        "source_filename": args.filename,
        "partition_count": len(partitions),
        "columns": sorted(columns),
    }
    (args.build_dir / "_index_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
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
        f"wrote {len(partitions)} metadata partitions to {args.output}",
        flush=True,
    )
    if not args.keep_source:
        shutil.rmtree(args.download_dir)
    shutil.rmtree(args.build_dir)


if __name__ == "__main__":
    main()
