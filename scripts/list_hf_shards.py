from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an hf:// tar shard list from a Hugging Face dataset."
    )
    parser.add_argument(
        "--repo",
        default="animetimm/danbooru-wdtagger-v4-w640-ws-full",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if args.start < 0:
        raise ValueError("--start cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    prefix = args.split.strip("/") + "/"
    files = sorted(
        filename
        for filename in api.list_repo_files(
            args.repo,
            repo_type="dataset",
        )
        if filename.startswith(prefix) and filename.lower().endswith(".tar")
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(files)
    files = files[args.start :]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(
            f"No tar files found in dataset {args.repo!r} under {prefix!r}"
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
