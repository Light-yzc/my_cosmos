from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_sd.config import load_yaml, require_section
from my_sd.autoencoders import WanVAEConfig
from my_sd.data.captions import DanbooruCaptionConfig, DanbooruCaptioner
from my_sd.data.latent_dataset import (
    BucketBatchSampler,
    LatentManifestDataset,
    collate_latents,
)
from my_sd.data.tar_stream import (
    StreamingLatentDataset,
    read_shard_list,
    shard_download_options,
)
from my_sd.data.raw_stream import RollingWanDataset
from my_sd.encoders import T5GemmaEncoder, TextEncoderConfig
from my_sd.models import CosmosDiT, CosmosDiTConfig
from my_sd.training.flow_matching import flow_matching_loss, make_flow_matching_batch
from my_sd.training.optimizers import build_optimizer
from my_sd.training.text_cache import apply_cfg_dropout, encode_text_windows
from my_sd.training.checkpoints import (
    AsyncCheckpointMirror,
    checkpoint_is_complete,
    prune_checkpoints,
    resolve_resume_path,
    update_latest,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported training precision: {name}") from error


def initialize_model(
    config: CosmosDiTConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> CosmosDiT:
    """Construct directly in the target dtype/device so every parameter is initialized."""
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = CosmosDiT(config)
    finally:
        torch.set_default_dtype(old_dtype)
    return model


def learning_rate_multiplier(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def save_checkpoint(
    model: CosmosDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    output_dir: Path,
    step: int,
    epoch: int,
    micro_step: int,
    data_cursor: dict[str, int] | None,
    *,
    mirror: AsyncCheckpointMirror | None = None,
    keep_last: int = 0,
) -> Path:
    checkpoint_dir = output_dir / f"step-{step:08d}"
    if checkpoint_is_complete(checkpoint_dir):
        update_latest(output_dir, checkpoint_dir)
        return checkpoint_dir

    temporary = output_dir / (
        f".{checkpoint_dir.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        state = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in model.state_dict().items()
        }
        save_file(state, str(temporary / "model.safetensors"))
        del state
        (temporary / "model_config.json").write_text(
            json.dumps(model.config.to_dict(), indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "micro_step": micro_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "data_cursor": data_cursor,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            temporary / "training_state.pt",
        )
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        temporary.replace(checkpoint_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    update_latest(output_dir, checkpoint_dir)
    if mirror is not None:
        mirror.submit(checkpoint_dir)
    else:
        prune_checkpoints(
            output_dir,
            keep_last,
            protected=(checkpoint_dir,),
        )
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    model: CosmosDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, Any]:
    weights_path = checkpoint_dir / "model.safetensors"
    training_path = checkpoint_dir / "training_state.pt"
    if not weights_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint_dir} must contain model.safetensors and training_state.pt"
        )
    weights = load_file(str(weights_path), device=str(device))
    model.load_state_dict(weights, strict=True)
    del weights
    state = torch.load(
        training_path,
        map_location=device,
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state.get("scaler", {}))
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(
        [value.cpu() for value in state["cuda_rng_state"]]
    )
    return state


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cosmos_08b_anime.yaml")
    parser.add_argument(
        "--resume",
        help="Checkpoint directory, training_state.pt path, or 'auto'.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rolling-block-size", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-mirror-dir")
    return parser


def apply_runtime_overrides(
    args: argparse.Namespace,
    data_config: dict[str, Any],
    train_config: dict[str, Any],
) -> None:
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        train_config["max_steps"] = args.max_steps
    if args.rolling_block_size is not None:
        if args.rolling_block_size < 1:
            raise ValueError("--rolling-block-size must be positive")
        data_config["rolling_block_size"] = args.rolling_block_size
    if args.output_dir is not None:
        train_config["output_dir"] = args.output_dir
    if args.checkpoint_mirror_dir is not None:
        train_config["checkpoint_mirror_dir"] = args.checkpoint_mirror_dir


def main() -> None:
    args = create_argument_parser().parse_args()
    raw = load_yaml(args.config)
    model_config = CosmosDiTConfig.from_dict(require_section(raw, "model"))
    text_config = TextEncoderConfig(**require_section(raw, "text_encoder"))
    data_config = require_section(raw, "data")
    train_config = require_section(raw, "train")
    apply_runtime_overrides(args, data_config, train_config)
    accumulation = int(train_config.get("gradient_accumulation_steps", 1))

    if not torch.cuda.is_available():
        raise RuntimeError("The full training configuration requires a CUDA GPU")
    device = torch.device("cuda")
    dtype = training_dtype(str(train_config.get("precision", "bfloat16")))
    seed = int(train_config.get("seed", 3407))
    seed_everything(seed)

    caption_config = DanbooruCaptionConfig(
        general_tag_dropout=float(data_config.get("general_tag_dropout", 0.10)),
        character_tag_dropout=float(data_config.get("character_tag_dropout", 0.02)),
        shuffle_general_tags=bool(data_config.get("shuffle_general_tags", True)),
        max_tags=int(data_config.get("max_tags", 128)),
        replace_underscores=bool(data_config.get("replace_underscores", True)),
    )
    captioner = DanbooruCaptioner(caption_config)
    backend = str(data_config.get("backend", "manifest"))
    sampler: BucketBatchSampler | None
    if backend == "manifest":
        dataset = LatentManifestDataset(
            data_config["manifest"],
            captioner=captioner,
            horizontal_flip_probability=float(
                data_config.get("horizontal_flip_probability", 0.0)
            ),
        )
        sampler = BucketBatchSampler(
            dataset,
            int(data_config.get("batch_size", 1)),
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
        workers = int(data_config.get("num_workers", 4))
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_latents,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
    elif backend == "streaming_tar":
        if int(data_config.get("batch_size", 1)) != 1:
            raise ValueError("The first streaming backend requires batch_size: 1")
        dataset = StreamingLatentDataset(
            read_shard_list(data_config["shard_list"]),
            cache_dir=data_config.get("cache_dir", "/content/shard_cache"),
            captioner=captioner,
            prefetch_shards=int(data_config.get("prefetch_shards", 2)),
            sample_shuffle_buffer=int(
                data_config.get("sample_shuffle_buffer", 256)
            ),
            delete_after_use=bool(data_config.get("delete_after_use", True)),
            **shard_download_options(data_config),
            seed=seed,
        )
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_latents,
            num_workers=0,
            pin_memory=True,
        )
    elif backend == "rolling_raw":
        if int(data_config.get("batch_size", 1)) != 1:
            raise ValueError("rolling_raw currently requires batch_size: 1")
        if int(train_config.get("text_cache_size", 0)) < 1:
            raise ValueError(
                "rolling_raw requires text_cache_size > 0 so T5 can be offloaded "
                "before each Wan encoding phase"
            )
        ratings = data_config.get("allowed_ratings")
        if ratings is not None and not isinstance(ratings, list):
            raise TypeError("data.allowed_ratings must be a YAML list")
        dataset = RollingWanDataset(
            read_shard_list(data_config["shard_list"]),
            wan_config=WanVAEConfig(
                wan_repo=str(data_config["wan_repo"]),
                checkpoint=str(data_config["vae_checkpoint"]),
                device="cuda",
                dtype=str(data_config.get("vae_dtype", "float16")),
                encoder_only=True,
            ),
            cache_dir=data_config.get("cache_dir", "/content/raw_cache"),
            captioner=captioner,
            resolution_stage=str(
                data_config.get("resolution_stage", "512")
            ),
            prefetch_shards=int(data_config.get("prefetch_shards", 1)),
            block_size=int(data_config.get("rolling_block_size", 2048)),
            encode_batch_size=int(data_config.get("encode_batch_size", 4)),
            accumulation_multiple=accumulation,
            max_upscale=float(data_config.get("max_upscale", 1.25)),
            allowed_ratings=ratings,
            require_metadata=bool(
                data_config.get("require_metadata", True)
            ),
            delete_after_use=bool(data_config.get("delete_after_use", True)),
            **shard_download_options(data_config),
            seed=seed,
        )
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_latents,
            num_workers=0,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Unknown data backend: {backend}")

    text_encoder = T5GemmaEncoder(text_config)
    if isinstance(dataset, RollingWanDataset):
        text_encoder.offload_to_cpu()
    model = initialize_model(model_config, device, dtype)
    model.train()
    optimizer = build_optimizer(model, train_config)
    max_steps = int(train_config["max_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step,
            warmup_steps=int(train_config.get("warmup_steps", 0)),
            total_steps=max_steps,
            minimum_ratio=float(train_config.get("min_learning_rate_ratio", 0.1)),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    cfg_dropout = float(train_config.get("cfg_dropout", 0.15))
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir_value = train_config.get("checkpoint_mirror_dir")
    keep_last = int(train_config.get("keep_last_checkpoints", 0))
    mirror = (
        AsyncCheckpointMirror(
            Path(str(mirror_dir_value)),
            keep_last=keep_last,
            keep_last_local=keep_last,
        )
        if mirror_dir_value
        else None
    )
    if mirror is not None:
        atexit.register(mirror.close)
    step = 0
    micro_step = 0
    epoch = 0
    last_data_cursor: dict[str, int] | None = None
    resume_path = resolve_resume_path(
        args.resume,
        output_dir,
        Path(str(mirror_dir_value)) if mirror_dir_value else None,
    )
    if resume_path is not None:
        restored = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        step = int(restored["step"])
        micro_step = int(restored.get("micro_step", step * accumulation))
        epoch = int(restored.get("epoch", 0))
        cursor = restored.get("data_cursor")
        if cursor and isinstance(
            dataset, (StreamingLatentDataset, RollingWanDataset)
        ):
            dataset.set_resume_cursor(**cursor)
            last_data_cursor = cursor
        print(f"resumed {resume_path} at optimizer step {step}")
    print(f"trainable parameters: {model.trainable_parameter_count():,}")
    sample_count = f"{len(dataset):,}" if hasattr(dataset, "__len__") else "streaming"
    print(f"samples: {sample_count}; gradient accumulation: {accumulation}")

    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        if isinstance(dataset, (StreamingLatentDataset, RollingWanDataset)):
            dataset.set_epoch(epoch)
        epoch_micro_start = micro_step
        text_window = int(train_config.get("text_cache_size", 0))
        if text_window:
            encoded_batches = encode_text_windows(
                loader,
                text_encoder,
                window_size=text_window,
                encoder_batch_size=int(
                    train_config.get("text_encode_batch_size", 16)
                ),
                cfg_dropout=cfg_dropout,
                cache_dtype=dtype,
                offload_between_windows=bool(
                    isinstance(dataset, RollingWanDataset)
                    or train_config.get(
                        "offload_text_encoder_between_windows", False
                    )
                ),
                cfg_seed=seed,
            )
        else:
            def encode_online() -> Any:
                for online_batch in loader:
                    captions = apply_cfg_dropout(
                        online_batch,
                        probability=cfg_dropout,
                        seed=seed,
                    )
                    states, mask = text_encoder.encode(captions)
                    yield online_batch, states, mask

            encoded_batches = encode_online()

        for batch, text_states, text_mask in encoded_batches:
            if "source_shard_index" in batch:
                last_data_cursor = {
                    "epoch": int(batch["stream_epoch"][0]),
                    "shard_index": int(batch["source_shard_index"][0]),
                    "sample_index": int(batch["source_sample_index"][0]),
                }
            clean = batch["latents"].to(
                device=device, dtype=dtype, non_blocking=True
            )
            text_states = text_states.to(device=device, dtype=dtype, non_blocking=True)
            text_mask = text_mask.to(device=device, non_blocking=True)
            flow = make_flow_matching_batch(
                clean,
                timestep_sampling=str(
                    train_config.get("timestep_sampling", "uniform")
                ),
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=dtype)
                if dtype != torch.float32
                else nullcontext()
            )
            with autocast:
                prediction = model(
                    flow.noisy_latents,
                    flow.timesteps,
                    text_states,
                    text_mask,
                )
                loss = flow_matching_loss(prediction, flow.target_velocity)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            if micro_step % accumulation:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_config.get("max_grad_norm", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % int(train_config.get("log_every", 10)) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"step={step} loss={loss.item():.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"seconds/step={elapsed / step:.2f}"
                )
            if step % int(train_config.get("save_every", 5000)) == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    output_dir,
                    step,
                    epoch,
                    micro_step,
                    last_data_cursor,
                    mirror=mirror,
                    keep_last=keep_last,
                )
            if step >= max_steps:
                break
        if step >= max_steps:
            break
        if micro_step == epoch_micro_start:
            raise RuntimeError(
                "The data backend yielded no trainable samples for this epoch"
            )
        epoch += 1

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        output_dir,
        step,
        epoch,
        micro_step,
        last_data_cursor,
        mirror=mirror,
        keep_last=keep_last,
    )
    if mirror is not None:
        mirror.close()


if __name__ == "__main__":
    main()
