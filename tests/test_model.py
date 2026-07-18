import torch
from torch.nn import functional as F

from my_sd.models import CosmosDiT, CosmosDiTConfig
from my_sd.models import cosmos_dit


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


def test_external_fa2_backend_uses_packed_qkv(monkeypatch) -> None:
    sdpa = cosmos_dit.SelfAttention2D(128, 4, backend="sdpa")
    fa2 = cosmos_dit.SelfAttention2D(128, 4, backend="flash_attn_2")
    fa2.load_state_dict(sdpa.state_dict())
    captured: dict[str, tuple[int, ...]] = {}

    def fake_flash_attention_2(qkv: torch.Tensor) -> torch.Tensor:
        captured["shape"] = tuple(qkv.shape)
        q, k, v = qkv.unbind(dim=2)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        return output.transpose(1, 2)

    monkeypatch.setattr(cosmos_dit, "flash_attention_2", fake_flash_attention_2)
    inputs = torch.randn(2, 15, 128)
    rope = (
        torch.ones(15, 32),
        torch.zeros(15, 32),
    )
    expected = sdpa(inputs, rope)
    actual = fa2(inputs, rope)
    assert captured["shape"] == (2, 15, 3, 4, 32)
    torch.testing.assert_close(actual, expected)


def test_unknown_self_attention_backend_is_rejected() -> None:
    config = tiny_config()
    config.self_attention_backend = "unknown"
    try:
        config.validate()
    except ValueError as error:
        assert "self_attention_backend" in str(error)
    else:
        raise AssertionError("Expected backend validation to fail")
