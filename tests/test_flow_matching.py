import torch

from my_sd.training.flow_matching import (
    flow_matching_loss,
    make_flow_matching_batch,
)


def test_rectified_flow_endpoints_and_target() -> None:
    clean = torch.tensor([[[[1.0]]], [[[2.0]]]])
    noise = torch.tensor([[[[5.0]]], [[[8.0]]]])
    timesteps = torch.tensor([0.0, 1.0])
    batch = make_flow_matching_batch(
        clean, noise=noise, timesteps=timesteps
    )
    assert torch.equal(batch.noisy_latents[0], clean[0])
    assert torch.equal(batch.noisy_latents[1], noise[1])
    assert torch.equal(batch.target_velocity, noise - clean)
    assert flow_matching_loss(batch.target_velocity, batch.target_velocity) == 0

