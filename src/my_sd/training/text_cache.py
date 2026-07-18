from __future__ import annotations

import random
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from my_sd.encoders.text import T5GemmaEncoder


def apply_cfg_dropout(
    batch: dict[str, Any],
    *,
    probability: float,
    seed: int | None = None,
) -> list[str]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("CFG dropout probability must be in [0, 1]")
    captions = list(batch["captions"])
    cursor_keys = (
        "stream_epoch",
        "source_shard_index",
        "source_sample_index",
    )
    deterministic = seed is not None and all(key in batch for key in cursor_keys)
    dropped: list[str] = []
    for index, caption in enumerate(captions):
        if deterministic:
            value = (
                int(seed)
                + int(batch["stream_epoch"][index]) * 1_000_003
                + int(batch["source_shard_index"][index]) * 9176
                + int(batch["source_sample_index"][index]) * 104729
            )
            should_drop = random.Random(value).random() < probability
        else:
            should_drop = random.random() < probability
        dropped.append("" if should_drop else caption)
    return dropped


def _pin_if_cuda(tensor: Tensor) -> Tensor:
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


@torch.no_grad()
def encode_text_windows(
    batches: Iterable[dict[str, Any]],
    text_encoder: T5GemmaEncoder,
    *,
    window_size: int,
    encoder_batch_size: int,
    cfg_dropout: float,
    cache_dtype: torch.dtype,
    offload_between_windows: bool = False,
    cfg_seed: int | None = None,
) -> Iterator[tuple[dict[str, Any], Tensor, Tensor]]:
    """
    Batch-encodes text for many micro-batches, then yields one training batch at
    a time. Hidden states live in CPU RAM only until their DiT step.
    """
    if window_size < 1 or encoder_batch_size < 1:
        raise ValueError("Text window and encoder batch sizes must be positive")
    iterator = iter(batches)
    while True:
        window: list[dict[str, Any]] = []
        caption_groups: list[list[str]] = []
        caption_count = 0
        while caption_count < window_size:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            captions = apply_cfg_dropout(
                batch,
                probability=cfg_dropout,
                seed=cfg_seed,
            )
            window.append(batch)
            caption_groups.append(captions)
            caption_count += len(captions)
        if not window:
            return

        text_started = time.monotonic()
        if offload_between_windows:
            text_encoder.move_to_configured_device()
        flat_captions = [caption for group in caption_groups for caption in group]
        print(
            f"[text] encoding {len(flat_captions):,} captions "
            f"in batches of {encoder_batch_size}",
            file=sys.stderr,
            flush=True,
        )
        encoded: list[Tensor] = []
        for start in range(0, len(flat_captions), encoder_batch_size):
            states, mask = text_encoder.encode(
                flat_captions[start : start + encoder_batch_size]
            )
            lengths = mask.sum(dim=1).tolist()
            for row, length in zip(states, lengths, strict=True):
                cached = row[: int(length)].to(
                    device="cpu", dtype=cache_dtype
                ).contiguous()
                encoded.append(_pin_if_cuda(cached))
            completed = min(start + encoder_batch_size, len(flat_captions))
            print(
                f"[text] {completed:,}/{len(flat_captions):,} captions encoded",
                file=sys.stderr,
                flush=True,
            )
        if offload_between_windows:
            text_encoder.offload_to_cpu()
            torch.cuda.empty_cache()
        print(
            f"[text] window ready in {time.monotonic() - text_started:.1f}s; "
            "starting DiT steps",
            file=sys.stderr,
            flush=True,
        )

        offset = 0
        for batch, captions in zip(window, caption_groups, strict=True):
            group = encoded[offset : offset + len(captions)]
            offset += len(captions)
            states = pad_sequence(group, batch_first=True)
            mask = torch.zeros(
                states.shape[:2],
                dtype=torch.bool,
                pin_memory=torch.cuda.is_available(),
            )
            for row, value in enumerate(group):
                mask[row, : value.shape[0]] = True
            yield batch, states, mask
