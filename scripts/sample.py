from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_sd.autoencoders import WanImageVAE, WanVAEConfig
from my_sd.config import load_yaml, require_section
from my_sd.encoders import T5GemmaEncoder, TextEncoderConfig
from my_sd.models import (
    CosmosDiTConfig,
    initialize_model,
    load_model_weights,
)
from my_sd.training.checkpoints import checkpoint_step, valid_checkpoints
from my_sd.training.precision import training_dtypes
from my_sd.training.sampling import sample_rectified_flow


def resolve_checkpoint(value: str | Path) -> Path:
    candidate = Path(value)
    if (candidate / "model.safetensors").is_file():
        return candidate
    checkpoints = valid_checkpoints(candidate)
    if checkpoints:
        return max(checkpoints, key=checkpoint_step)
    if candidate.is_file() and candidate.name == "model.safetensors":
        return candidate.parent
    raise FileNotFoundError(
        f"{candidate} is neither a checkpoint nor a directory containing one"
    )


def image_from_tensor(tensor: torch.Tensor) -> Image.Image:
    value = (
        tensor.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.asarray(value), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a training checkpoint and decode it with Wan2.2 VAE."
    )
    parser.add_argument("--config", default="configs/colab_l4_fa2_deepghs.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, default=Path("sample.png"))
    args = parser.parse_args()

    if args.width < 32 or args.height < 32:
        raise ValueError("width and height must be at least 32")
    if args.width % 32 or args.height % 32:
        raise ValueError("width and height must be divisible by 32")
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling requires a CUDA GPU")

    raw = load_yaml(args.config)
    text_config = TextEncoderConfig(**require_section(raw, "text_encoder"))
    data_config = require_section(raw, "data")
    train_config = require_section(raw, "train")
    checkpoint = resolve_checkpoint(args.checkpoint)
    saved_model_config = checkpoint / "model_config.json"
    if saved_model_config.is_file():
        model_values = json.loads(saved_model_config.read_text(encoding="utf-8"))
    else:
        model_values = require_section(raw, "model")
    model_values["gradient_checkpointing"] = False
    model_config = CosmosDiTConfig.from_dict(model_values)
    compute_dtype, parameter_dtype = training_dtypes(train_config)
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    print("[sample] encoding positive and negative prompts", flush=True)
    text_encoder = T5GemmaEncoder(text_config)
    text_encoder.move_to_configured_device()
    states, mask = text_encoder.encode([args.negative_prompt, args.prompt])
    states = states.to(device=device, dtype=compute_dtype).clone()
    mask = mask.to(device=device).clone()
    negative_states, positive_states = states.chunk(2)
    negative_mask, positive_mask = mask.chunk(2)
    text_encoder.offload_to_cpu()
    del text_encoder
    torch.cuda.empty_cache()

    print(f"[sample] loading DiT from {checkpoint}", flush=True)
    model = initialize_model(model_config, device, parameter_dtype)
    load_model_weights(model, checkpoint)
    model.eval()
    latents = torch.randn(
        1,
        model_config.latent_channels,
        args.height // 16,
        args.width // 16,
        device=device,
        dtype=compute_dtype,
    )
    autocast = (
        torch.autocast("cuda", dtype=compute_dtype)
        if compute_dtype != torch.float32
        else nullcontext()
    )
    print(
        f"[sample] {args.solver} flow solve: {args.steps} steps, "
        f"CFG {args.guidance_scale:g}",
        flush=True,
    )
    with autocast:
        latents = sample_rectified_flow(
            model,
            latents,
            positive_states=positive_states,
            positive_mask=positive_mask,
            negative_states=negative_states,
            negative_mask=negative_mask,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            solver=args.solver,
        )
    latents = latents.to(device="cpu", dtype=torch.float32)
    model.to("cpu")
    del model, states, mask, positive_states, negative_states
    del positive_mask, negative_mask
    torch.cuda.empty_cache()

    print("[sample] loading Wan2.2 decoder", flush=True)
    vae = WanImageVAE(
        WanVAEConfig(
            wan_repo=str(data_config["wan_repo"]),
            checkpoint=str(data_config["vae_checkpoint"]),
            device="cuda",
            dtype=str(data_config.get("vae_dtype", "float16")),
            encoder_only=False,
        )
    )
    decoded = vae.decode_images(latents)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_from_tensor(decoded).save(args.output)
    print(f"[sample] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
