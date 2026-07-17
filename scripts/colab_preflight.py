from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_sd.training.preflight import run_colab_preflight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a streaming training config before loading model weights."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-bitsandbytes", action="store_true")
    args = parser.parse_args()

    report = run_colab_preflight(
        args.config,
        cwd=ROOT,
        require_cuda=not args.no_cuda,
        check_assets=not args.skip_assets,
        check_bitsandbytes=not args.skip_bitsandbytes,
    )
    labels = {"ok": "OK", "warning": "WARN", "error": "ERROR"}
    for check in report.checks:
        print(f"[{labels[check.level]}] {check.name}: {check.message}")
    if report.ok:
        print("preflight passed")
        return 0
    print(f"preflight failed with {len(report.errors)} error(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
