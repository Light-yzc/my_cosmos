from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_sd.config import load_yaml, require_section
from my_sd.models import CosmosDiT, CosmosDiTConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cosmos_08b_anime.yaml")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    config = CosmosDiTConfig.from_dict(require_section(raw, "model"))
    with torch.device("meta"):
        model = CosmosDiT(config)

    sections = {
        "patch_embed": model.patch_embed,
        "text_adapter": model.text_adapter,
        "timestep": model.timestep,
        "transformer_blocks": model.blocks,
        "output_head": torch.nn.ModuleList(
            [model.final_norm, model.final_modulation, model.output_projection]
        ),
    }
    print(f"configuration: {config.depth} x {config.hidden_size}, {config.num_heads} heads")
    for name, module in sections.items():
        count = sum(parameter.numel() for parameter in module.parameters())
        print(f"{name:20s} {count:>15,d}")
    print(f"{'trainable total':20s} {model.trainable_parameter_count():>15,d}")


if __name__ == "__main__":
    main()

