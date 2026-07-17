from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class TextEncoderConfig:
    model_id: str = "google/t5gemma-2-270m-270m"
    revision: str = "main"
    max_length: int = 256
    dtype: str = "bfloat16"
    device: str = "cuda"
    cpu_offload: bool = False
    encoder_only_checkpoint: bool = False


def _torch_dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return values[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


class T5GemmaEncoder(nn.Module):
    """
    Loads T5Gemma-2, discards its decoder, and keeps the frozen encoder only.

    The full checkpoint is loaded on CPU first, so decoder weights never occupy
    GPU memory. The steady-state GPU component is the roughly 270M encoder.
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from transformers.models.t5gemma2.modeling_t5gemma2 import (
                T5Gemma2TextEncoder,
            )
        except ImportError as error:
            raise RuntimeError("Install transformers to use T5GemmaEncoder") from error

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            use_fast=True,
        )
        requested_dtype = _torch_dtype(config.dtype)
        self.compute_dtype = requested_dtype
        self.configured_device = torch.device(config.device)
        load_dtype = torch.float32 if config.device == "cpu" else requested_dtype
        if config.encoder_only_checkpoint:
            self.encoder = T5Gemma2TextEncoder.from_pretrained(
                config.model_id,
                revision=config.revision,
                dtype=load_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            full_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model_id,
                revision=config.revision,
                dtype=load_dtype,
                low_cpu_mem_usage=True,
            )
            multimodal_encoder = full_model.get_encoder()
            self.encoder = multimodal_encoder.text_model
            del multimodal_encoder
            del full_model
        hidden_size = int(self.encoder.config.hidden_size)
        if hidden_size != 640:
            raise ValueError(
                f"Expected the T5Gemma-2 270M encoder hidden size 640, got {hidden_size}"
            )
        self.encoder.eval().requires_grad_(False)
        self.execution_device = torch.device(
            "cpu" if config.cpu_offload else config.device
        )
        if self.execution_device.type == "cpu":
            self.encoder.to(device=self.execution_device, dtype=torch.float32)
        else:
            self.encoder.to(device=self.execution_device, dtype=requested_dtype)

    def move_to_configured_device(self) -> None:
        dtype = (
            torch.float32
            if self.configured_device.type == "cpu"
            else self.compute_dtype
        )
        self.encoder.to(device=self.configured_device, dtype=dtype)
        self.execution_device = self.configured_device

    def offload_to_cpu(self) -> None:
        self.encoder.to(device="cpu", dtype=self.compute_dtype)
        self.execution_device = torch.device("cpu")

    @torch.inference_mode()
    def encode(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        tokens = {
            name: value.to(self.execution_device)
            for name, value in tokens.items()
            if name in {"input_ids", "attention_mask"}
        }
        output = self.encoder(**tokens, return_dict=True)
        return output.last_hidden_state.detach(), tokens["attention_mask"].bool()

    def forward(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        return self.encode(prompts)
