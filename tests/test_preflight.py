from pathlib import Path

from my_sd.training.preflight import run_colab_preflight


def _write_config(tmp_path: Path, *, block_size: int = 32) -> Path:
    wan_repo = tmp_path / "Wan2.2"
    wan_module = wan_repo / "wan" / "modules" / "vae2_2.py"
    wan_module.parent.mkdir(parents=True)
    wan_module.write_text("# stub\n", encoding="utf-8")
    vae = tmp_path / "Wan2.2_VAE.pth"
    vae.write_bytes(b"stub")
    text_encoder = tmp_path / "text-encoder"
    text_encoder.mkdir()
    raw_tar = tmp_path / "raw.tar"
    raw_tar.write_bytes(b"stub")
    shard_list = tmp_path / "shards.txt"
    shard_list.write_text(str(raw_tar) + "\n", encoding="utf-8")
    config = tmp_path / "rolling.yaml"
    config.write_text(
        f"""
model: {{}}
text_encoder:
  model_id: {text_encoder.as_posix()}
data:
  backend: rolling_raw
  shard_list: {shard_list.as_posix()}
  cache_dir: {(tmp_path / 'cache').as_posix()}
  prefetch_shards: 1
  rolling_block_size: {block_size}
  wan_repo: {wan_repo.as_posix()}
  vae_checkpoint: {vae.as_posix()}
train:
  gradient_accumulation_steps: 8
  text_cache_size: 16
  optimizer: adamw8bit
  allow_optimizer_fallback: false
  precision: float16
  output_dir: {(tmp_path / 'checkpoints').as_posix()}
  checkpoint_mirror_dir: {(tmp_path / 'mirror').as_posix()}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def test_preflight_accepts_complete_rolling_configuration(tmp_path) -> None:
    report = run_colab_preflight(
        _write_config(tmp_path),
        cwd=tmp_path,
        require_cuda=False,
        check_bitsandbytes=False,
    )
    assert report.ok, report.errors
    assert any(check.name == "rolling block" for check in report.checks)
    assert any(
        check.name == "gradient checkpointing" for check in report.checks
    )


def test_preflight_rejects_block_outside_optimizer_boundary(tmp_path) -> None:
    report = run_colab_preflight(
        _write_config(tmp_path, block_size=30),
        cwd=tmp_path,
        require_cuda=False,
        check_bitsandbytes=False,
    )
    assert not report.ok
    assert any(check.name == "rolling block" for check in report.errors)
