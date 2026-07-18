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
from safetensors.torch import save_file
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
from my_sd.models import (
    CosmosDiT,
    CosmosDiTConfig,
    initialize_model,
    load_model_weights,
)
from my_sd.training.flow_matching import flow_matching_loss, make_flow_matching_batch
from my_sd.training.optimizers import build_optimizer
from my_sd.training.precision import grad_scaler_enabled, training_dtypes
from my_sd.training.text_cache import apply_cfg_dropout, encode_text_windows
from my_sd.training.checkpoints import (
    AsyncCheckpointMirror,
    checkpoint_is_complete,
    prune_checkpoints,
    replace_directory,
    resolve_resume_path,
    update_latest,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cuda_backends() -> None:
    """Enable fast, numerically safe CUDA paths across supported PyTorch versions."""
    try:
        # PyTorch 2.9+ spelling. TF32 only affects explicit FP32 matmuls; the
        # L4 recipe primarily computes in BF16.
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except (AttributeError, RuntimeError):
        # Compatibility with the project's PyTorch >=2.5 lower bound.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


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
        replace_directory(temporary, checkpoint_dir)
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
    load_model_weights(model, weights_path)
    state = torch.load(
        training_path,
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(state.pop("optimizer"))
    scheduler.load_state_dict(state.pop("scheduler"))
    scaler.load_state_dict(state.pop("scaler", {}))
    random.setstate(state.pop("python_rng_state"))
    np.random.set_state(state.pop("numpy_rng_state"))
    torch.set_rng_state(state.pop("torch_rng_state").cpu())
    torch.cuda.set_rng_state_all(
        [value.cpu() for value in state.pop("cuda_rng_state")]
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


def initialize_wandb(
    raw_config: dict[str, Any],
    train_config: dict[str, Any],
) -> Any | None:
    wandb_config = train_config.get("wandb", {})
    if not isinstance(wandb_config, dict):
        raise TypeError("train.wandb must be a mapping")
    if not bool(wandb_config.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging is enabled; run `uv sync --extra train --extra fa2`"
        ) from error
    mode = str(wandb_config.get("mode", "online"))
    if mode == "online" and not (
        os.environ.get("WANDB_API_KEY")
        or Path.home().joinpath(".netrc").is_file()
    ):
        raise RuntimeError(
            "W&B online logging is enabled but no login was found. Add "
            "WANDB_API_KEY to Colab Secrets or run `uv run wandb login`."
        )
    state_root_value = train_config.get(
        "checkpoint_mirror_dir",
        train_config.get("output_dir", "checkpoints"),
    )
    state_root = Path(str(state_root_value))
    state_root.mkdir(parents=True, exist_ok=True)
    run_id_path = state_root / "wandb-run-id.txt"
    if run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = ""
    if not run_id:
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
    run = wandb.init(
        project=str(wandb_config.get("project", "cosmos-anime")),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("name"),
        tags=list(wandb_config.get("tags", [])),
        mode=mode,
        config=raw_config,
        id=run_id,
        resume="allow",
    )
    run.define_metric("train/optimizer_step")
    run.define_metric("*", step_metric="train/optimizer_step")
    return run


def _move_nested_tensors(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _move_nested_tensors(item, device)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _move_nested_tensors(item, device)
        return value
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device) for item in value)
    return value


def move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> None:
    for state in optimizer.state.values():
        _move_nested_tensors(state, device)


def main() -> None:
    args = create_argument_parser().parse_args()
    raw = load_yaml(args.config)
    model_config = CosmosDiTConfig.from_dict(require_section(raw, "model"))
    text_config = TextEncoderConfig(**require_section(raw, "text_encoder"))
    data_config = require_section(raw, "data")
    train_config = require_section(raw, "train")
    apply_runtime_overrides(args, data_config, train_config)
    wandb_run = initialize_wandb(raw, train_config)
    if wandb_run is not None:
        atexit.register(wandb_run.finish)
    accumulation = int(train_config.get("gradient_accumulation_steps", 1))
    train_batch_size = int(data_config.get("batch_size", 1))

    if not torch.cuda.is_available():
        raise RuntimeError("The full training configuration requires a CUDA GPU")
    device = torch.device("cuda")
    dtype, parameter_dtype = training_dtypes(train_config)
    seed = int(train_config.get("seed", 3407))
    seed_everything(seed)
    configure_cuda_backends()

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
        if train_batch_size < 1:
            raise ValueError("rolling_raw data.batch_size must be positive")
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
            train_batch_size=train_batch_size,
            decode_prefetch=int(data_config.get("decode_prefetch", 16)),
            accumulation_multiple=accumulation * train_batch_size,
            max_upscale=float(data_config.get("max_upscale", 1.25)),
            allowed_ratings=ratings,
            require_metadata=bool(
                data_config.get("require_metadata", True)
            ),
            metadata_index_dir=data_config.get("metadata_index_dir"),
            delete_after_use=bool(data_config.get("delete_after_use", True)),
            **shard_download_options(data_config),
            seed=seed,
        )
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            collate_fn=collate_latents,
            num_workers=0,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Unknown data backend: {backend}")

    text_encoder = T5GemmaEncoder(text_config)
    if isinstance(dataset, RollingWanDataset):
        text_encoder.offload_to_cpu()
    model = initialize_model(model_config, device, parameter_dtype)
    model.train()
    optimizer = build_optimizer(model, train_config)
    if isinstance(dataset, RollingWanDataset):
        train_state_on_gpu = True

        def before_wan_encode() -> None:
            nonlocal train_state_on_gpu
            if not train_state_on_gpu:
                return
            started = time.perf_counter()
            model.to("cpu")
            move_optimizer_state(optimizer, "cpu")
            train_state_on_gpu = False
            torch.cuda.empty_cache()
            print(
                f"[phase] DiT+optimizer -> CPU in "
                f"{time.perf_counter() - started:.1f}s; Wan VAE may use GPU",
                flush=True,
            )

        def before_dit_train() -> None:
            nonlocal train_state_on_gpu
            if train_state_on_gpu:
                return
            started = time.perf_counter()
            model.to(device)
            move_optimizer_state(optimizer, device)
            train_state_on_gpu = True
            torch.cuda.empty_cache()
            print(
                f"[phase] DiT+optimizer -> GPU in "
                f"{time.perf_counter() - started:.1f}s; starting training",
                flush=True,
            )

        dataset.set_phase_hooks(
            before_encode=before_wan_encode,
            before_train=before_dit_train,
        )
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
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=grad_scaler_enabled(dtype, parameter_dtype),
    )
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
        del restored
    print(f"trainable parameters: {model.trainable_parameter_count():,}")
    print(
        f"compute dtype: {dtype}; parameter dtype: {parameter_dtype}; "
        f"gradient scaler: {scaler.is_enabled()}"
    )
    print(f"self-attention backend: {model_config.self_attention_backend}")
    sample_count = f"{len(dataset):,}" if hasattr(dataset, "__len__") else "streaming"
    print(
        f"samples: {sample_count}; microbatch: {train_batch_size}; "
        f"gradient accumulation: {accumulation}; effective batch: "
        f"{train_batch_size * accumulation}"
    )

    optimizer.zero_grad(set_to_none=True)
    initial_step = step
    started = time.perf_counter()
    last_batch_finished = started
    last_optimizer_finished = started
    input_wait_window = 0.0
    loss_sum_window = torch.zeros((), device=device, dtype=torch.float32)
    loss_count_window = 0
    last_grad_norm = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
            batch_received = time.perf_counter()
            input_wait_window += max(batch_received - last_batch_finished, 0.0)
            if "source_shard_index" in batch:
                last_data_cursor = {
                    "epoch": int(batch["stream_epoch"][0]),
                    "shard_index": int(
                        batch.get(
                            "resume_shard_index",
                            batch["source_shard_index"],
                        )[0]
                    ),
                    "sample_index": int(
                        batch.get(
                            "resume_sample_index",
                            batch["source_sample_index"],
                        )[0]
                    ),
                }
            clean = batch["latents"].to(
                device=device, dtype=dtype, non_blocking=True
            ).clone()
            text_states = text_states.to(
                device=device, dtype=dtype, non_blocking=True
            ).clone()
            text_mask = text_mask.to(
                device=device, non_blocking=True
            ).clone()
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
            loss_sum_window.add_(loss.detach().float())
            loss_count_window += 1
            micro_step += 1
            if micro_step % accumulation:
                last_batch_finished = time.perf_counter()
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_config.get("max_grad_norm", 1.0))
            )
            last_grad_norm = float(grad_norm.detach().float().item())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % int(train_config.get("log_every", 10)) == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now = time.perf_counter()
                elapsed = time.perf_counter() - started
                runtime_steps = max(step - initial_step, 1)
                step_wall = max(now - last_optimizer_finished, 1e-6)
                input_fraction = min(input_wait_window / step_wall, 1.0)
                mean_loss = float(
                    (loss_sum_window / max(loss_count_window, 1)).item()
                )
                peak_gib = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                reserved_gib = (
                    torch.cuda.memory_reserved() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                peak_reserved_gib = (
                    torch.cuda.max_memory_reserved() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                samples_per_second = (
                    accumulation * train_batch_size / step_wall
                )
                cursor_text = ""
                if last_data_cursor is not None:
                    cursor_text = (
                        f" shard={last_data_cursor['shard_index']} "
                        f"sample={last_data_cursor['sample_index']}"
                    )
                print(
                    f"step={step} loss={mean_loss:.5f} "
                    f"grad_norm={last_grad_norm:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"seconds/step={elapsed / runtime_steps:.2f} "
                    f"last_step={step_wall:.2f}s "
                    f"samples/s={samples_per_second:.2f} "
                    f"input_wait={input_wait_window:.2f}s "
                    f"input_wait_ratio={input_fraction:.1%} "
                    f"cuda={peak_gib:.1f}GiB_peak/"
                    f"{reserved_gib:.1f}GiB_reserved"
                    f"{cursor_text}",
                    flush=True,
                )
                if wandb_run is not None:
                    metrics: dict[str, float | int] = {
                        "train/optimizer_step": step,
                        "train/micro_step": micro_step,
                        "train/loss": mean_loss,
                        "train/gradient_norm": last_grad_norm,
                        "train/learning_rate": float(
                            scheduler.get_last_lr()[0]
                        ),
                        "performance/seconds_per_step_average": (
                            elapsed / runtime_steps
                        ),
                        "performance/log_window_seconds": step_wall,
                        "performance/samples_per_second": samples_per_second,
                        "performance/input_wait_seconds": input_wait_window,
                        "performance/input_wait_ratio": input_fraction,
                        "system/cuda_peak_allocated_gib": peak_gib,
                        "system/cuda_reserved_gib": reserved_gib,
                        "system/cuda_peak_reserved_gib": peak_reserved_gib,
                    }
                    if last_data_cursor is not None:
                        metrics.update(
                            {
                                "data/epoch": last_data_cursor["epoch"],
                                "data/shard_index": last_data_cursor[
                                    "shard_index"
                                ],
                                "data/sample_index": last_data_cursor[
                                    "sample_index"
                                ],
                            }
                        )
                    wandb_run.log(metrics, step=step)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                last_optimizer_finished = now
                input_wait_window = 0.0
                loss_sum_window.zero_()
                loss_count_window = 0
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
            last_batch_finished = time.perf_counter()
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
    if wandb_run is not None:
        wandb_run.finish()
        atexit.unregister(wandb_run.finish)


if __name__ == "__main__":
    main()
