from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any


def list_repo_tar_files(
    api: Any,
    *,
    repo: str,
    split: str,
    revision: str,
    retries: int,
) -> list[str]:
    prefix = split.strip("/") + "/"
    for attempt in range(retries):
        try:
            return sorted(
                filename
                for filename in api.list_repo_files(
                    repo,
                    repo_type="dataset",
                    revision=revision,
                )
                if filename.startswith(prefix)
                and filename.lower().endswith(".tar")
            )
        except Exception:
            if attempt + 1 >= retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(
        description="Create an hf:// tar shard list from a Hugging Face dataset."
    )
    parser.add_argument(
        "--repo",
        default="animetimm/danbooru-wdtagger-v4-w640-ws-full",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    if args.start < 0:
        raise ValueError("--start cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.retries < 1:
        raise ValueError("--retries must be positive")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = list_repo_tar_files(
        api,
        repo=args.repo,
        split=args.split,
        revision=args.revision,
        retries=args.retries,
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(files)
    files = files[args.start :]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(
            f"No tar files found in dataset {args.repo!r} "
            f"under {args.split.strip('/') + '/'!r}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = [
        f"hf://datasets/{args.repo}/{filename}"
        for filename in files
    ]
    args.output.write_text("\n".join(values) + "\n", encoding="utf-8")
    print(f"wrote {len(values)} shards to {args.output}")


if __name__ == "__main__":
    main()
