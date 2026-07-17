import torch

from my_sd.models import CosmosDiT, CosmosDiTConfig


def tiny_config(*, checkpointing: bool = False) -> CosmosDiTConfig:
    return CosmosDiTConfig(
        latent_channels=8,
        patch_size=2,
        hidden_size=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        context_dim=96,
        text_input_dim=64,
        text_adapter_depth=1,
        text_adapter_heads=4,
        adaln_rank=16,
        rope_extrapolation=1.0,
        gradient_checkpointing=checkpointing,
    )


def test_forward_preserves_non_square_latent_shape() -> None:
    model = CosmosDiT(tiny_config())
    latents = torch.randn(2, 8, 12, 20)
    timesteps = torch.tensor([0.1, 0.9])
    text = torch.randn(2, 11, 64)
    mask = torch.tensor(
        [[1] * 11, [1] * 7 + [0] * 4],
        dtype=torch.bool,
    )
    output = model(latents, timesteps, text, mask)
    assert output.shape == latents.shape
    assert torch.count_nonzero(output) == 0


def test_activation_checkpoint_backward() -> None:
    model = CosmosDiT(tiny_config(checkpointing=True)).train()
    latents = torch.randn(1, 8, 8, 12)
    target = torch.randn_like(latents)
    output = model(
        latents,
        torch.tensor([0.5]),
        torch.randn(1, 6, 64),
        torch.ones(1, 6, dtype=torch.bool),
    )
    loss = (output - target).square().mean()
    loss.backward()
    assert model.output_projection.weight.grad is not None


def test_invalid_latent_channel_count_is_rejected() -> None:
    model = CosmosDiT(tiny_config())
    try:
        model(
            torch.randn(1, 7, 8, 8),
            torch.tensor([0.5]),
            torch.randn(1, 4, 64),
        )
    except ValueError as error:
        assert "Expected 8 latent channels" in str(error)
    else:
        raise AssertionError("Expected a channel validation error")

