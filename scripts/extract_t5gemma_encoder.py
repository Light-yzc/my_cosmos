from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time conversion of the official T5Gemma-2 package to its "
            "270M text encoder only."
        )
    )
    parser.add_argument(
        "--model-id",
        default="google/t5gemma-2-270m-270m",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    text_encoder = model.get_encoder().text_model
    if int(text_encoder.config.hidden_size) != 640:
        raise RuntimeError(
            f"Unexpected encoder hidden size: {text_encoder.config.hidden_size}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    text_encoder.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        use_fast=True,
    )
    tokenizer.save_pretrained(args.output)
    parameter_count = sum(parameter.numel() for parameter in text_encoder.parameters())
    print(f"saved {parameter_count:,} encoder parameters to {args.output}")


if __name__ == "__main__":
    main()

