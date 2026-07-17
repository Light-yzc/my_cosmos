from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(slots=True)
class FlowMatchingBatch:
    noisy_latents: Tensor
    target_velocity: Tensor
    timesteps: Tensor
    noise: Tensor


def sample_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    method: str = "uniform",
) -> Tensor:
    if method == "uniform":
        return torch.rand(batch_size, device=device)
    if method == "logit_normal":
        return torch.randn(batch_size, device=device).sigmoid()
    raise ValueError(f"Unknown timestep sampling method: {method}")


def make_flow_matching_batch(
    clean_latents: Tensor,
    *,
    timesteps: Tensor | None = None,
    noise: Tensor | None = None,
    timestep_sampling: str = "uniform",
) -> FlowMatchingBatch:
    """
    Straight rectified-flow path with data at t=0 and Gaussian noise at t=1.

    x_t = (1-t) * x_data + t * epsilon
    velocity target = epsilon - x_data
    """
    batch = clean_latents.shape[0]
    if timesteps is None:
        timesteps = sample_timesteps(
            batch, device=clean_latents.device, method=timestep_sampling
        )
    if timesteps.shape != (batch,):
        raise ValueError(f"timesteps must have shape ({batch},)")
    if noise is None:
        noise = torch.randn_like(clean_latents)
    if noise.shape != clean_latents.shape:
        raise ValueError("noise and clean_latents must have identical shapes")
    broadcast_t = timesteps.to(dtype=clean_latents.dtype).view(
        batch, *([1] * (clean_latents.ndim - 1))
    )
    noisy = torch.lerp(clean_latents, noise, broadcast_t)
    target = noise - clean_latents
    return FlowMatchingBatch(noisy, target, timesteps, noise)


def flow_matching_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(prediction.float(), target.float())

