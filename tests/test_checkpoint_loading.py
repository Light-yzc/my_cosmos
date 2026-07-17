import torch

from my_sd.models import CosmosDiTConfig, initialize_model
from scripts.train import load_checkpoint, save_checkpoint


def _small_config() -> CosmosDiTConfig:
    return CosmosDiTConfig(
        latent_channels=4,
        patch_size=2,
        hidden_size=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        context_dim=24,
        text_input_dim=16,
        text_adapter_depth=1,
        text_adapter_heads=4,
        adaln_rank=8,
        gradient_checkpointing=False,
    )


def test_checkpoint_load_restores_model_and_returns_only_metadata(tmp_path) -> None:
    model = initialize_model(_small_config(), torch.device("cpu"), torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    checkpoint = save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        tmp_path,
        step=3,
        epoch=1,
        micro_step=12,
        data_cursor={"epoch": 1, "shard_index": 2, "sample_index": 10},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    restored = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cpu"),
    )

    assert set(restored) == {"step", "epoch", "micro_step", "data_cursor"}
    assert restored["step"] == 3
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])
