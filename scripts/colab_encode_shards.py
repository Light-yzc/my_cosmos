from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_sd.autoencoders import WanImageVAE, WanVAEConfig
from my_sd.data.buckets import (
    DEFAULT_BUCKETS,
    HIGH_RES_BUCKETS,
    LOW_RES_BUCKETS,
    choose_bucket,
)
from my_sd.data.raw_stream import iter_raw_tar, prepare_image
from my_sd.data.tar_stream import (
    AsyncShardPrefetcher,
    LatentTarWriter,
    read_shard_list,
)


def output_name(source: str, stage: str) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:20]
    return f"latent/{stage}/latent-{digest}.tar"


def sample_id(source: str, key: str) -> str:
    return hashlib.sha1(f"{source}\0{key}".encode("utf-8")).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Colab producer: prefetch raw image tar shards, encode permanent Wan "
            "latents, and optionally append them to a Hugging Face dataset repo."
        )
    )
    parser.add_argument("--shard-list", required=True, type=Path)
    parser.add_argument("--output-dir", default="/content/latent_output", type=Path)
    parser.add_argument("--cache-dir", default="/content/raw_cache", type=Path)
    parser.add_argument("--wan-repo", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument(
        "--resolution-stage", choices=("512", "768", "1024"), default="512"
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--prefetch-shards", type=int, default=2)
    parser.add_argument("--download-retries", type=int, default=4)
    parser.add_argument("--minimum-free-gb", type=float, default=2.0)
    parser.add_argument("--max-cache-gb", type=float, default=8.0)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--max-upscale", type=float, default=1.25)
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="Encode image-only samples. By default samples without tags/caption are skipped.",
    )
    parser.add_argument(
        "--ratings",
        help="Optional comma-separated rating whitelist, for example g,s.",
    )
    parser.add_argument("--upload-repo")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--delete-after-upload", action="store_true")
    args = parser.parse_args()

    buckets = {
        "512": LOW_RES_BUCKETS,
        "768": DEFAULT_BUCKETS,
        "1024": HIGH_RES_BUCKETS,
    }[args.resolution_stage]
    allowed_ratings = (
        {value.strip().lower() for value in args.ratings.split(",") if value.strip()}
        if args.ratings
        else None
    )
    sources = read_shard_list(args.shard_list)
    remote_files: set[str] = set()
    api = None
    if args.upload_repo:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(
            args.upload_repo,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        remote_files = set(
            api.list_repo_files(args.upload_repo, repo_type="dataset")
        )

    pending_sources: list[str] = []
    for source in sources:
        path_in_repo = output_name(source, args.resolution_stage)
        local_output = args.output_dir / path_in_repo
        if path_in_repo in remote_files:
            continue
        if local_output.is_file():
            if api is not None:
                api.upload_file(
                    path_or_fileobj=local_output,
                    path_in_repo=path_in_repo,
                    repo_id=args.upload_repo,
                    repo_type="dataset",
                    commit_message=f"Resume {args.resolution_stage} latent upload",
                )
                if args.delete_after_upload:
                    local_output.unlink()
            continue
        pending_sources.append(source)
    if not pending_sources:
        print("all source shards are already encoded")
        return

    vae = WanImageVAE(
        WanVAEConfig(
            wan_repo=args.wan_repo,
            checkpoint=args.vae_checkpoint,
            device="cuda",
            dtype=args.dtype,
            encoder_only=True,
        )
    )
    prefetcher = AsyncShardPrefetcher(
        pending_sources,
        args.cache_dir,
        prefetch=args.prefetch_shards,
        delete_after_use=True,
        retries=args.download_retries,
        minimum_free_bytes=int(args.minimum_free_gb * 1024**3),
        max_cache_bytes=int(args.max_cache_gb * 1024**3),
    )

    for source, local_raw in zip(pending_sources, prefetcher, strict=True):
        path_in_repo = output_name(source, args.resolution_stage)
        destination = args.output_dir / path_in_repo
        encoded_count = 0
        skipped_count = 0
        with LatentTarWriter(destination) as writer:
            bucket_buffers: dict[
                str, list[tuple[str, torch.Tensor, dict[str, Any]]]
            ] = {}

            def flush(bucket_key: str) -> None:
                nonlocal encoded_count, skipped_count
                items = list(bucket_buffers.get(bucket_key, []))
                if not items:
                    return
                bucket_buffers[bucket_key].clear()
                pixels = torch.stack([item[1] for item in items])
                try:
                    latents = vae.encode_images(pixels)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    try:
                        latents = torch.cat(
                            [
                                vae.encode_images(pixel.unsqueeze(0))
                                for pixel in pixels
                            ],
                            dim=0,
                        )
                    except Exception as error:
                        skipped_count += len(items)
                        print(
                            f"skip {len(items)} images in {bucket_key}: {error}",
                            file=sys.stderr,
                        )
                        return
                except Exception as error:
                    skipped_count += len(items)
                    print(
                        f"skip {len(items)} images in {bucket_key}: {error}",
                        file=sys.stderr,
                    )
                    return
                for (identifier, _, record), latent in zip(
                    items, latents, strict=True
                ):
                    writer.add(identifier, latent, record)
                    encoded_count += 1

            samples = iter_raw_tar(
                local_raw,
                require_metadata=not args.allow_missing_metadata,
            )
            for key, image, metadata in tqdm(
                samples, desc=Path(path_in_repo).name, unit="image"
            ):
                try:
                    if (
                        allowed_ratings is not None
                        and str(metadata.get("rating", "")).strip().lower()
                        not in allowed_ratings
                    ):
                        skipped_count += 1
                        continue
                    bucket = choose_bucket(image.width, image.height, buckets)
                    required_scale = max(
                        bucket.width / image.width, bucket.height / image.height
                    )
                    if required_scale > args.max_upscale:
                        skipped_count += 1
                        continue
                    original_size = (image.width, image.height)
                    pixels, crop_box = prepare_image(image, bucket)
                    record = dict(metadata)
                    record.update(
                        {
                            "bucket": bucket.key,
                            "original_size": list(original_size),
                            "crop_box": list(crop_box),
                            "source_shard": source,
                            "source_member": key,
                            "vae": "Wan2.2_TI2V_5B_f16c48",
                        }
                    )
                    buffer = bucket_buffers.setdefault(bucket.key, [])
                    buffer.append((sample_id(source, key), pixels, record))
                    if len(buffer) >= args.encode_batch_size:
                        flush(bucket.key)
                except Exception as error:
                    skipped_count += 1
                    print(f"skip {key}: {error}", file=sys.stderr)
            for bucket_key in list(bucket_buffers):
                flush(bucket_key)

        if encoded_count == 0:
            destination.unlink(missing_ok=True)
            print(f"{source}: no encodable images; skipped {skipped_count}")
            continue
        if api is not None:
            api.upload_file(
                path_or_fileobj=destination,
                path_in_repo=path_in_repo,
                repo_id=args.upload_repo,
                repo_type="dataset",
                commit_message=f"Add {args.resolution_stage} latent shard",
            )
            if args.delete_after_upload:
                destination.unlink()
        print(
            f"{source}: encoded {encoded_count}, skipped {skipped_count}, "
            f"saved {path_in_repo}"
        )


if __name__ == "__main__":
    main()
