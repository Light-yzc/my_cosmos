from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_sd.data.buckets import (
    DEFAULT_BUCKETS,
    HIGH_RES_BUCKETS,
    LOW_RES_BUCKETS,
    choose_bucket,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_sidecar(image_path: Path) -> dict[str, object]:
    json_path = image_path.with_suffix(".json")
    text_path = image_path.with_suffix(".txt")
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"{json_path} must contain a JSON object")
        return value
    if text_path.is_file():
        return {"general_tags": text_path.read_text(encoding="utf-8").strip()}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an image/tag JSONL manifest from Danbooru-style sidecars."
    )
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--resolution-stage",
        choices=("512", "768", "1024"),
        default="768",
    )
    parser.add_argument("--max-upscale", type=float, default=1.25)
    args = parser.parse_args()
    buckets = {
        "512": LOW_RES_BUCKETS,
        "768": DEFAULT_BUCKETS,
        "1024": HIGH_RES_BUCKETS,
    }[args.resolution_stage]
    images = sorted(
        path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as output:
        for path in tqdm(images, desc="Scanning"):
            try:
                with Image.open(path) as image:
                    width, height = image.size
                bucket = choose_bucket(width, height, buckets)
                required_scale = max(bucket.width / width, bucket.height / height)
                if required_scale > args.max_upscale:
                    skipped += 1
                    continue
                record: dict[str, object] = {
                    "image_path": str(path.resolve()),
                    "original_width": width,
                    "original_height": height,
                    "bucket": bucket.key,
                }
                record.update(load_sidecar(path))
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            except Exception as error:
                skipped += 1
                print(f"skip {path}: {error}", file=sys.stderr)
    print(f"wrote {kept} records to {args.output}; skipped {skipped}")


if __name__ == "__main__":
    main()
