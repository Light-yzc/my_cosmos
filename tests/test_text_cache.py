import torch

from my_sd.training.text_cache import apply_cfg_dropout, encode_text_windows


class FakeTextEncoder:
    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []
        self.moves = 0
        self.offloads = 0

    def encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoded_batches.append(prompts)
        lengths = [max(1, len(prompt.split())) for prompt in prompts]
        maximum = max(lengths)
        states = torch.zeros(len(prompts), maximum, 4)
        mask = torch.zeros(len(prompts), maximum, dtype=torch.bool)
        for row, length in enumerate(lengths):
            states[row, :length] = row + 1
            mask[row, :length] = True
        return states, mask

    def move_to_configured_device(self) -> None:
        self.moves += 1

    def offload_to_cpu(self) -> None:
        self.offloads += 1


def test_text_window_batches_encoder_work() -> None:
    encoder = FakeTextEncoder()
    batches = [
        {"captions": ["one two"], "latents": torch.zeros(1, 1, 1, 1)},
        {"captions": ["three"], "latents": torch.zeros(1, 1, 1, 1)},
        {"captions": ["four five six"], "latents": torch.zeros(1, 1, 1, 1)},
    ]
    output = list(
        encode_text_windows(
            batches,
            encoder,  # type: ignore[arg-type]
            window_size=2,
            encoder_batch_size=2,
            cfg_dropout=0.0,
            cache_dtype=torch.float16,
            offload_between_windows=True,
        )
    )
    assert len(output) == 3
    assert encoder.encoded_batches == [["one two", "three"], ["four five six"]]
    assert encoder.moves == 2
    assert encoder.offloads == 2
    assert output[0][1].dtype == torch.float16


def test_cfg_dropout_is_stable_for_stream_cursor() -> None:
    batch = {
        "captions": ["a", "b"],
        "stream_epoch": [2, 2],
        "source_shard_index": [4, 4],
        "source_sample_index": [10, 11],
    }
    first = apply_cfg_dropout(batch, probability=0.5, seed=123)
    second = apply_cfg_dropout(batch, probability=0.5, seed=123)
    assert first == second
