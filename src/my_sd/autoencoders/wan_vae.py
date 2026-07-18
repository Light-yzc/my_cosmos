from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
import torch
from torch import Tensor


@dataclass(slots=True)
class WanVAEConfig:
    wan_repo: str
    checkpoint: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    latent_channels: int = 48
    encoder_only: bool = True


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported Wan VAE dtype: {name}") from error


def _load_wan_vae_type(module_path: Path):
    """Load vae2_2.py without executing Wan's dependency-heavy package init."""
    digest = hashlib.sha1(
        str(module_path).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    module_name = f"_my_sd_wan_vae2_2_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Wan VAE module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    try:
        return getattr(module, "Wan2_2_VAE")
    except AttributeError as error:
        raise ImportError(f"{module_path} does not define Wan2_2_VAE") from error


class WanImageVAE:
    """Thin static-image wrapper around Wan2.2 TI2V-5B's official f16c48 VAE."""

    spatial_compression = 16

    def __init__(self, config: WanVAEConfig) -> None:
        repo = Path(config.wan_repo).resolve()
        checkpoint = Path(config.checkpoint).resolve()
        module_path = repo / "wan" / "modules" / "vae2_2.py"
        if not module_path.is_file():
            raise FileNotFoundError(
                f"{repo} is not a Wan2.2 source checkout (missing wan/modules/vae2_2.py)"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        vae_type = _load_wan_vae_type(module_path)
        target_device = config.device
        target_dtype = config.dtype
        load_device = "cpu" if config.encoder_only else target_device
        self.config = replace(config)
        self.vae = vae_type(
            z_dim=config.latent_channels,
            vae_pth=str(checkpoint),
            dtype=_dtype(target_dtype),
            device=load_device,
        )
        if config.encoder_only:
            self.vae.model.decoder = torch.nn.Identity()
            self.vae.model.conv2 = torch.nn.Identity()
        self.move_to(target_device, target_dtype)

    def move_to(self, device: str, dtype: str | None = None) -> None:
        dtype_name = dtype or self.config.dtype
        compute_dtype = _dtype(dtype_name)
        self.vae.model.to(device=device, dtype=compute_dtype)
        self.vae.scale = [
            value.to(device=device, dtype=compute_dtype)
            if isinstance(value, Tensor)
            else value
            for value in self.vae.scale
        ]
        self.vae.device = device
        self.vae.dtype = compute_dtype
        self.config.device = device
        self.config.dtype = dtype_name

    def offload_to_cpu(self) -> None:
        self.move_to("cpu", self.config.dtype)

    @torch.inference_mode()
    def encode_images(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError("Wan images must have height and width divisible by 16")
        videos = images.to(
            self.config.device, non_blocking=True
        ).unsqueeze(2)
        device_type = torch.device(self.config.device).type
        with torch.autocast(device_type=device_type, dtype=_dtype(self.config.dtype)):
            latents = self.vae.model.encode(videos, self.vae.scale).float()
        if latents.shape[2] != 1:
            raise RuntimeError(f"Expected one latent frame, got shape {tuple(latents.shape)}")
        return latents.squeeze(2)

    @torch.inference_mode()
    def decode_images(self, latents: Tensor) -> Tensor:
        if self.config.encoder_only:
            raise RuntimeError(
                "This Wan VAE was loaded encoder-only. Set encoder_only=False for decoding."
            )
        if latents.ndim != 4 or latents.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"latents must have shape [B,{self.config.latent_channels},H,W]"
            )
        videos = latents.to(self.config.device).unsqueeze(2)
        device_type = torch.device(self.config.device).type
        with torch.autocast(device_type=device_type, dtype=_dtype(self.config.dtype)):
            decoded = self.vae.model.decode(videos, self.vae.scale).float()
        return decoded.clamp_(-1, 1).squeeze(2)
