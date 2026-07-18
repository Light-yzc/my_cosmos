from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

VelocityModel = Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor]


def classifier_free_velocity(
    model: VelocityModel,
    latents: Tensor,
    timesteps: Tensor,
    *,
    positive_states: Tensor,
    positive_mask: Tensor,
    negative_states: Tensor,
    negative_mask: Tensor,
    guidance_scale: float,
) -> Tensor:
    """Evaluate positive/negative prompts together and apply standard CFG."""
    if latents.shape[0] != 1:
        raise ValueError("The single-GPU sampler currently supports batch size 1")
    if guidance_scale < 0:
        raise ValueError("guidance_scale cannot be negative")
    model_latents = torch.cat((latents, latents), dim=0)
    model_timesteps = torch.cat((timesteps, timesteps), dim=0)
    context = torch.cat((negative_states, positive_states), dim=0)
    context_mask = torch.cat((negative_mask, positive_mask), dim=0)
    unconditional, conditional = model(
        model_latents,
        model_timesteps,
        context,
        context_mask,
    ).chunk(2)
    return unconditional + guidance_scale * (conditional - unconditional)


@torch.no_grad()
def sample_rectified_flow(
    model: VelocityModel,
    latents: Tensor,
    *,
    positive_states: Tensor,
    positive_mask: Tensor,
    negative_states: Tensor,
    negative_mask: Tensor,
    steps: int = 28,
    guidance_scale: float = 5.0,
    solver: str = "heun",
) -> Tensor:
    """
    Integrate the learned velocity field from Gaussian noise at t=1 to data at
    t=0. Heun is the default quality-oriented second-order solver.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    schedule = torch.linspace(
        1.0,
        0.0,
        steps + 1,
        device=latents.device,
        dtype=torch.float32,
    )

    def velocity(value: Tensor, timestep: Tensor) -> Tensor:
        batch_timestep = timestep.expand(value.shape[0])
        return classifier_free_velocity(
            model,
            value,
            batch_timestep,
            positive_states=positive_states,
            positive_mask=positive_mask,
            negative_states=negative_states,
            negative_mask=negative_mask,
            guidance_scale=guidance_scale,
        )

    current = latents
    for index in range(steps):
        timestep = schedule[index]
        next_timestep = schedule[index + 1]
        delta = (next_timestep - timestep).to(dtype=current.dtype)
        first = velocity(current, timestep)
        if solver == "euler" or index + 1 == steps:
            current = current + delta * first
            continue
        predictor = current + delta * first
        second = velocity(predictor, next_timestep)
        current = current + delta * (first + second) * 0.5
    return current
