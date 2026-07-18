import torch

from my_sd.autoencoders.wan_vae import WanImageVAE, WanVAEConfig


def test_encoder_only_vae_is_trimmed_on_cpu_before_target_move(
    tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "Wan2.2"
    module_path = repo / "wan" / "modules" / "vae2_2.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# test stub\n", encoding="utf-8")
    checkpoint = tmp_path / "vae.pth"
    checkpoint.write_bytes(b"stub")

    class FakeModel:
        def __init__(self) -> None:
            self.decoder = torch.nn.Linear(1, 1)
            self.conv2 = torch.nn.Linear(1, 1)
            self.moves: list[tuple[str, torch.dtype, bool]] = []

        def to(self, *, device: str, dtype: torch.dtype) -> "FakeModel":
            trimmed = isinstance(self.decoder, torch.nn.Identity) and isinstance(
                self.conv2, torch.nn.Identity
            )
            self.moves.append((device, dtype, trimmed))
            return self

    class FakeWanVAE:
        def __init__(
            self,
            *,
            z_dim: int,
            vae_pth: str,
            dtype: torch.dtype,
            device: str,
        ) -> None:
            self.constructor_device = device
            self.model = FakeModel()
            self.scale = [1.0, 2.0]

    monkeypatch.setattr(
        "my_sd.autoencoders.wan_vae._load_wan_vae_type",
        lambda _: FakeWanVAE,
    )
    wrapper = WanImageVAE(
        WanVAEConfig(
            wan_repo=str(repo),
            checkpoint=str(checkpoint),
            device="cuda",
            dtype="float16",
            encoder_only=True,
        )
    )

    assert wrapper.vae.constructor_device == "cpu"
    assert wrapper.vae.model.moves == [("cuda", torch.float16, True)]
    assert wrapper.config.device == "cuda"


def test_wan_vae_loader_bypasses_package_init(tmp_path) -> None:
    repo = tmp_path / "Wan2.2"
    package = repo / "wan"
    module_path = package / "modules" / "vae2_2.py"
    module_path.parent.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('package init must not run')\n",
        encoding="utf-8",
    )
    module_path.write_text(
        "class Wan2_2_VAE:\n    pass\n",
        encoding="utf-8",
    )

    from my_sd.autoencoders.wan_vae import _load_wan_vae_type

    loaded = _load_wan_vae_type(module_path)
    assert loaded.__name__ == "Wan2_2_VAE"
