from __future__ import annotations

import importlib
import os
import shutil
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from my_sd.config import load_yaml, require_section
from my_sd.data.tar_stream import read_shard_list

CheckLevel = Literal["ok", "warning", "error"]


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    level: CheckLevel
    name: str
    message: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def errors(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if check.level == "error")

    @property
    def ok(self) -> bool:
        return not self.errors


def _existing_parent(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def _is_remote_shard(source: str) -> bool:
    if source.startswith("hf://"):
        return True
    return urllib.parse.urlparse(source).scheme in {"http", "https"}


def _path_value(value: object, cwd: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else cwd / path


def run_colab_preflight(
    config_path: str | Path,
    *,
    cwd: str | Path | None = None,
    require_cuda: bool = True,
    check_assets: bool = True,
    check_bitsandbytes: bool = True,
) -> PreflightReport:
    config_file = Path(config_path).resolve()
    working_dir = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    raw = load_yaml(config_file)
    data = require_section(raw, "data")
    train = require_section(raw, "train")
    text = require_section(raw, "text_encoder")
    checks: list[PreflightCheck] = []

    backend = str(data.get("backend", "manifest"))
    checks.append(PreflightCheck("ok", "backend", backend))
    if backend not in {"rolling_raw", "streaming_tar"}:
        checks.append(
            PreflightCheck(
                "error",
                "backend",
                "Colab streaming requires data.backend=rolling_raw or streaming_tar",
            )
        )

    accumulation = int(train.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        checks.append(
            PreflightCheck("error", "gradient accumulation", "must be positive")
        )
    elif backend == "rolling_raw":
        block_size = int(data.get("rolling_block_size", 0))
        if block_size < 1 or block_size % accumulation:
            checks.append(
                PreflightCheck(
                    "error",
                    "rolling block",
                    f"rolling_block_size={block_size} must be a positive multiple "
                    f"of gradient_accumulation_steps={accumulation}",
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "ok",
                    "rolling block",
                    f"{block_size} samples; optimizer boundary every {accumulation}",
                )
            )
        if int(data.get("prefetch_shards", 1)) != 1:
            checks.append(
                PreflightCheck(
                    "error",
                    "raw prefetch",
                    "rolling_raw requires prefetch_shards=1",
                )
            )
        if int(train.get("text_cache_size", 0)) < 1:
            checks.append(
                PreflightCheck(
                    "error",
                    "text cache",
                    "rolling_raw requires train.text_cache_size > 0",
                )
            )

    shard_list_value = data.get("shard_list")
    if not shard_list_value:
        checks.append(PreflightCheck("error", "shards", "data.shard_list is missing"))
        sources: list[str] = []
    else:
        shard_list = _path_value(shard_list_value, working_dir)
        try:
            sources = read_shard_list(shard_list)
        except (OSError, ValueError) as error:
            checks.append(PreflightCheck("error", "shards", str(error)))
            sources = []
        else:
            missing_local = [
                source
                for source in sources
                if not _is_remote_shard(source)
                and not _path_value(source, working_dir).is_file()
            ]
            if missing_local:
                checks.append(
                    PreflightCheck(
                        "error",
                        "shards",
                        f"local shard does not exist: {missing_local[0]}",
                    )
                )
            else:
                checks.append(
                    PreflightCheck(
                        "ok", "shards", f"{len(sources)} source shard(s)"
                    )
                )
            if any(source.startswith("hf://") for source in sources) and not os.environ.get(
                "HF_TOKEN"
            ):
                checks.append(
                    PreflightCheck(
                        "warning",
                        "Hugging Face token",
                        "HF_TOKEN is unset; gated datasets will fail to download",
                    )
                )

    if backend == "rolling_raw" and check_assets:
        wan_repo = _path_value(data.get("wan_repo", ""), working_dir)
        wan_module = wan_repo / "wan" / "modules" / "vae2_2.py"
        if not wan_module.is_file():
            checks.append(
                PreflightCheck(
                    "error", "Wan2.2 source", f"missing {wan_module}"
                )
            )
        else:
            checks.append(PreflightCheck("ok", "Wan2.2 source", str(wan_repo)))
        vae_checkpoint = _path_value(data.get("vae_checkpoint", ""), working_dir)
        if not vae_checkpoint.is_file():
            checks.append(
                PreflightCheck(
                    "error", "Wan VAE checkpoint", f"missing {vae_checkpoint}"
                )
            )
        else:
            checks.append(
                PreflightCheck("ok", "Wan VAE checkpoint", str(vae_checkpoint))
            )

    model_id = str(text.get("model_id", ""))
    model_path = _path_value(model_id, working_dir)
    is_local_model = Path(model_id).is_absolute() or model_id.startswith(".")
    if check_assets and is_local_model and not model_path.is_dir():
        checks.append(
            PreflightCheck("error", "text encoder", f"missing {model_path}")
        )
    else:
        checks.append(PreflightCheck("ok", "text encoder", model_id))

    optimizer = str(train.get("optimizer", "adamw")).lower()
    strict_8bit = optimizer == "adamw8bit" and not bool(
        train.get("allow_optimizer_fallback", True)
    )
    if strict_8bit and check_bitsandbytes:
        try:
            importlib.import_module("bitsandbytes")
        except (ImportError, RuntimeError) as error:
            checks.append(
                PreflightCheck(
                    "error", "8-bit optimizer", f"bitsandbytes unavailable: {error}"
                )
            )
        else:
            checks.append(PreflightCheck("ok", "8-bit optimizer", "bitsandbytes"))

    precision = str(train.get("precision", "bfloat16")).lower()
    parameter_precision = str(
        train.get(
            "parameter_precision",
            "float32" if precision in {"float16", "fp16"} else precision,
        )
    ).lower()
    if require_cuda:
        if not torch.cuda.is_available():
            checks.append(PreflightCheck("error", "CUDA", "CUDA GPU not available"))
        else:
            properties = torch.cuda.get_device_properties(0)
            memory_gib = properties.total_memory / 1024**3
            checks.append(
                PreflightCheck(
                    "ok",
                    "CUDA",
                    f"{properties.name}; {memory_gib:.1f} GiB",
                )
            )
            requests_bf16 = any(
                value in {"bfloat16", "bf16"}
                for value in (precision, parameter_precision)
            )
            if requests_bf16 and not torch.cuda.is_bf16_supported():
                checks.append(
                    PreflightCheck(
                        "error",
                        "precision",
                        "configuration requests BF16 but this GPU lacks BF16 support",
                    )
                )
            else:
                checks.append(
                    PreflightCheck(
                        "ok",
                        "precision",
                        f"compute={precision}; parameters={parameter_precision}",
                    )
                )

    cache_dir = _path_value(data.get("cache_dir", "/content/raw_cache"), working_dir)
    cache_parent = _existing_parent(cache_dir)
    if cache_parent is None:
        checks.append(
            PreflightCheck("error", "cache disk", f"no existing parent for {cache_dir}")
        )
    else:
        free_bytes = shutil.disk_usage(cache_parent).free
        minimum = int(float(data.get("minimum_free_gb", 0.0)) * 1024**3)
        if free_bytes < minimum:
            checks.append(
                PreflightCheck(
                    "error",
                    "cache disk",
                    f"{free_bytes / 1024**3:.1f} GiB free; "
                    f"minimum_free_gb={minimum / 1024**3:.1f}",
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "ok", "cache disk", f"{free_bytes / 1024**3:.1f} GiB free"
                )
            )

    mirror_value = train.get("checkpoint_mirror_dir")
    if mirror_value:
        mirror = _path_value(mirror_value, working_dir)
        drive_root = Path("/content/drive/MyDrive")
        if str(mirror).startswith(str(drive_root)) and not drive_root.is_dir():
            checks.append(
                PreflightCheck(
                    "error",
                    "Google Drive",
                    "/content/drive/MyDrive is not mounted",
                )
            )
        else:
            checks.append(PreflightCheck("ok", "checkpoint mirror", str(mirror)))

    return PreflightReport(tuple(checks))
