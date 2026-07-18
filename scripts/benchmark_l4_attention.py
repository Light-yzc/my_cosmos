from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "gpu_smoke.py"
SECONDS_PATTERN = re.compile(r"seconds/iteration=([0-9.]+)")


def benchmark(
    backend: str,
    *,
    config: Path,
    pixel_size: str,
    warmup: int,
    iterations: int,
    optimizer: str,
) -> float:
    command = [
        sys.executable,
        str(SMOKE),
        "--config",
        str(config),
        "--pixel-size",
        pixel_size,
        "--precision",
        "bfloat16",
        "--parameter-precision",
        "bfloat16",
        "--text-length",
        "192",
        "--self-attention-backend",
        backend,
        "--sdpa-backend",
        "auto",
        "--gradient-checkpointing",
        "--optimizer",
        optimizer,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"\n--- {backend} ---")
    print(completed.stdout.rstrip())
    if completed.returncode:
        raise RuntimeError(
            f"gpu_smoke failed for {backend} with exit code "
            f"{completed.returncode}"
        )
    match = SECONDS_PATTERN.search(completed.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse gpu_smoke output for {backend}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch SDPA and external FlashAttention-2 on an L4."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/colab_l4_fa2_24gb.yaml"),
    )
    parser.add_argument("--pixel-size", default="768x768")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--optimizer",
        choices=("none", "adamw8bit"),
        default="adamw8bit",
    )
    args = parser.parse_args()
    try:
        import flash_attn  # noqa: F401
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "The benchmark requires a working external flash-attn install"
        ) from error

    sdpa = benchmark(
        "sdpa",
        config=args.config,
        pixel_size=args.pixel_size,
        warmup=args.warmup,
        iterations=args.iterations,
        optimizer=args.optimizer,
    )
    fa2 = benchmark(
        "flash_attn_2",
        config=args.config,
        pixel_size=args.pixel_size,
        warmup=args.warmup,
        iterations=args.iterations,
        optimizer=args.optimizer,
    )
    speedup = sdpa / fa2
    print("\n--- comparison ---")
    print(f"sdpa_seconds={sdpa:.4f}")
    print(f"fa2_seconds={fa2:.4f}")
    print(f"fa2_speedup={speedup:.3f}x ({(speedup - 1.0) * 100.0:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
