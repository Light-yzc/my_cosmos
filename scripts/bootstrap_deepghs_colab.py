from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def ensure_drive() -> None:
    target = Path("/content/drive/MyDrive")
    if target.is_dir():
        return
    try:
        from google.colab import drive
    except ImportError as error:
        raise RuntimeError(
            "Google Drive is not mounted and this is not a Colab runtime"
        ) from error
    drive.mount("/content/drive")
    if not target.is_dir():
        raise RuntimeError("Google Drive mount did not create /content/drive/MyDrive")


def ensure_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from huggingface_hub import get_token

        token = (get_token() or "").strip()
    except ImportError:
        token = ""
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Accept the DeepGHS dataset terms, then set "
            "`os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')`."
        )
    os.environ["HF_TOKEN"] = token
    return token


def import_colab_secret(name: str) -> None:
    if os.environ.get(name):
        return
    try:
        from google.colab import userdata

        value = userdata.get(name)
    except Exception:
        return
    if value:
        os.environ[name] = value


def verify_deepghs_access(token: str) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    repo = "deepghs/danbooru2024-webp-4Mpixel"
    print("verifying gated DeepGHS image access...", flush=True)
    try:
        probe = Path(
            hf_hub_download(
                repo_id=repo,
                filename="images/0000.json",
                repo_type="dataset",
                local_dir="/content/deepghs_access_check",
                token=token,
            )
        )
    except HfHubHTTPError as error:
        status = getattr(error.response, "status_code", None)
        if status in {401, 403}:
            raise RuntimeError(
                "HF_TOKEN is valid for public metadata but its account cannot "
                "read the gated DeepGHS images. Open "
                "https://huggingface.co/datasets/deepghs/"
                "danbooru2024-webp-4Mpixel while logged into the SAME account, "
                "accept the access terms, then rerun this command."
            ) from error
        raise
    print(f"DeepGHS image access verified: {probe.name}", flush=True)


def ensure_wan_source(path: Path) -> None:
    module = path / "wan" / "modules" / "vae2_2.py"
    if module.is_file():
        print(f"reusing Wan2.2 source: {path}")
        return
    if path.exists():
        raise RuntimeError(f"{path} exists but does not contain {module.relative_to(path)}")
    run(
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/Wan-Video/Wan2.2.git",
        str(path),
    )


def ensure_vae(models: Path, token: str) -> None:
    from huggingface_hub import hf_hub_download

    checkpoint = models / "Wan2.2_VAE.pth"
    if checkpoint.is_file():
        print(f"reusing Wan VAE: {checkpoint}")
        return
    models.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        filename="Wan2.2_VAE.pth",
        local_dir=models,
        token=token,
    )


def ensure_text_encoder(models: Path) -> None:
    output = models / "t5gemma2-270m-encoder"
    if (output / "config.json").is_file():
        print(f"reusing text encoder: {output}")
        return
    run(
        sys.executable,
        "scripts/extract_t5gemma_encoder.py",
        "--output",
        str(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare every DeepGHS/Wan/T5 asset and launch L4 training."
    )
    parser.add_argument(
        "--config",
        default="configs/colab_l4_fa2_deepghs.yaml",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/cosmos"),
    )
    parser.add_argument("--smoke-shards", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    if args.smoke_shards is not None and args.smoke_shards < 1:
        raise ValueError("--smoke-shards must be positive")

    ensure_drive()
    import_colab_secret("HF_TOKEN")
    import_colab_secret("WANDB_API_KEY")
    token = ensure_token()
    verify_deepghs_access(token)
    models = Path("/content/models")
    args.drive_root.mkdir(parents=True, exist_ok=True)

    ensure_wan_source(Path("/content/Wan2.2"))
    ensure_vae(models, token)
    ensure_text_encoder(models)

    run(
        sys.executable,
        "scripts/prepare_deepghs_metadata.py",
        "--download-dir",
        "/content/deepghs_metadata_source",
        "--build-dir",
        "/content/deepghs_metadata_build",
        "--output",
        str(args.drive_root / "deepghs_metadata"),
    )
    shard_command = [
        sys.executable,
        "scripts/list_hf_shards.py",
        "--repo",
        "deepghs/danbooru2024-webp-4Mpixel",
        "--split",
        "images",
        "--output",
        "/content/deepghs_raw_shards.txt",
    ]
    if args.smoke_shards is not None:
        shard_command.extend(["--limit", str(args.smoke_shards)])
    run(*shard_command)
    run(sys.executable, "scripts/colab_preflight.py", "--config", args.config)

    if args.prepare_only:
        print("all assets are ready; training was not started")
        return
    run(sys.executable, "scripts/train.py", "--config", args.config)


if __name__ == "__main__":
    main()
