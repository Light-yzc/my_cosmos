from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_sd.config import load_yaml, require_section
from my_sd.models import CosmosDiTConfig, initialize_model
from my_sd.training.precision import grad_scaler_enabled, precision_dtype


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    try:
        height, width = (int(part) for part in normalized.split("x", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("size must look like 512x768") from error
    if height < 32 or width < 32 or height % 32 or width % 32:
        raise argparse.ArgumentTypeError(
            "pixel height and width must be positive multiples of 32"
        )
    return height, width


def sdpa_context(name: str):
    if name == "auto":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "math": SDPBackend.MATH,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    return sdpa_kernel(backends[name])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CUDA forward/backward throughput and peak-memory smoke test."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--pixel-size", type=parse_size, default=(512, 512))
    parser.add_argument("--precision", default="float16")
    parser.add_argument(
        "--parameter-precision",
        default="auto",
        help="Parameter/master-weight dtype; auto uses FP32 for FP16 compute.",
    )
    parser.add_argument("--depth", type=int)
    parser.add_argument("--text-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "math", "efficient", "flash", "cudnn"),
        default="auto",
    )
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument(
        "--optimizer",
        choices=("none", "adamw8bit", "adamw"),
        default="none",
        help="Include an optimizer step and its state in the memory benchmark.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    raw = load_yaml(args.config)
    values = dict(require_section(raw, "model"))
    if args.depth is not None:
        values["depth"] = args.depth
    if args.gradient_checkpointing is not None:
        values["gradient_checkpointing"] = args.gradient_checkpointing
    config = CosmosDiTConfig.from_dict(values)
    dtype = precision_dtype(args.precision)
    parameter_dtype = (
        torch.float32
        if args.parameter_precision == "auto" and dtype == torch.float16
        else dtype
        if args.parameter_precision == "auto"
        else precision_dtype(args.parameter_precision)
    )
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support native BF16; use --precision fp16")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = initialize_model(config, device=device, dtype=parameter_dtype).train()
    pixel_height, pixel_width = args.pixel_size
    latent_height, latent_width = pixel_height // 16, pixel_width // 16
    latents = torch.randn(
        1,
        config.latent_channels,
        latent_height,
        latent_width,
        device=device,
        dtype=dtype,
    )
    text = torch.randn(
        1,
        args.text_length,
        config.text_input_dim,
        device=device,
        dtype=dtype,
    )
    text_mask = torch.ones(
        1,
        args.text_length,
        device=device,
        dtype=torch.bool,
    )
    text_mask[:, -max(1, args.text_length // 8) :] = False
    timesteps = torch.rand(1, device=device)
    target = torch.randn_like(latents)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=grad_scaler_enabled(dtype, parameter_dtype),
    )
    optimizer = None
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            foreach=False,
        )

    def run_iteration() -> float:
        if optimizer is None:
            model.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast("cuda", dtype=dtype)
            if dtype != torch.float32
            else nullcontext()
        )
        # The SDPA selection must also wrap backward: activation checkpointing
        # recomputes attention there, and changing kernels would change saved
        # tensor metadata.
        with sdpa_context(args.sdpa_backend):
            with autocast:
                output = model(latents, timesteps, text, text_mask)
                loss = (output - target).square().mean()
            if not args.forward_only:
                scaler.scale(loss).backward()
                if optimizer is not None:
                    scaler.step(optimizer)
                    scaler.update()
        return float(loss.detach())

    try:
        for _ in range(args.warmup):
            run_iteration()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss = 0.0
        for _ in range(args.iterations):
            loss = run_iteration()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    except torch.OutOfMemoryError:
        print("CUDA OOM")
        print(torch.cuda.memory_summary(abbreviated=True))
        return 2

    parameters = sum(value.numel() for value in model.parameters())
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(
        f"parameters={parameters:,} depth={config.depth} "
        f"pixel={pixel_height}x{pixel_width} latent={latent_height}x{latent_width}"
    )
    print(
        f"compute_precision={args.precision} parameter_dtype={parameter_dtype} "
        f"scaler={scaler.is_enabled()} checkpointing={config.gradient_checkpointing} "
        f"sdpa={args.sdpa_backend} backward={not args.forward_only} "
        f"optimizer={args.optimizer}"
    )
    print(
        f"seconds/iteration={elapsed / args.iterations:.4f} "
        f"iterations/second={args.iterations / elapsed:.4f} loss={loss:.6f}"
    )
    print(f"peak_allocated_gib={allocated:.3f} peak_reserved_gib={reserved:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
