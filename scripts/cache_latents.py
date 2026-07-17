from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_sd.autoencoders import WanImageVAE, WanVAEConfig
from my_sd.data.buckets import cover_resize_and_center_crop, parse_bucket


def image_tensor(path: Path, bucket_key: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    bucket = parse_bucket(bucket_key)
    with Image.open(path) as source:
        image = source.convert("RGB")
        original_size = image.size
        resize_size, crop_box = cover_resize_and_center_crop(
            image.width, image.height, bucket
        )
        image = image.resize(resize_size, Image.Resampling.LANCZOS)
        image = image.crop(crop_box)
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)
    return tensor, (*original_size, *crop_box)


def sample_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode center-cropped static images with Wan2.2's f16c48 VAE."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--wan-repo", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = args.output_dir / "latents"
    latent_dir.mkdir(exist_ok=True)
    output_manifest = args.output_dir / "manifest.jsonl"
    vae = WanImageVAE(
        WanVAEConfig(
            wan_repo=args.wan_repo,
            checkpoint=args.vae_checkpoint,
            device=args.device,
            dtype=args.dtype,
        )
    )

    written: list[dict[str, object]] = []
    for record in tqdm(records, desc="Encoding Wan latents"):
        image_path = Path(str(record["image_path"])).resolve()
        identifier = sample_id(image_path)
        destination = latent_dir / f"{identifier}.safetensors"
        tensor, geometry = image_tensor(image_path, str(record["bucket"]))
        if args.overwrite or not destination.is_file():
            latent = vae.encode_images(tensor.unsqueeze(0))[0]
            save_file(
                {"latent": latent.to(device="cpu", dtype=torch.float16).contiguous()},
                str(destination),
                metadata={
                    "vae": "Wan2.2_TI2V_5B_f16c48",
                    "bucket": str(record["bucket"]),
                },
            )
        cached = dict(record)
        cached["sample_id"] = identifier
        cached["latent_path"] = str(destination.resolve())
        cached["preprocess_geometry"] = list(geometry)
        written.append(cached)

    temporary = output_manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in written:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_manifest)
    print(f"wrote {len(written)} cached records to {output_manifest}")


if __name__ == "__main__":
    main()

