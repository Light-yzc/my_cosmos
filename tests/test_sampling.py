import torch

from my_sd.training.sampling import sample_rectified_flow


class ConstantVelocity:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del timesteps, text_states, text_mask
        self.calls += 1
        return torch.full_like(latents, self.value)


def _condition() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool)


def test_euler_flow_integrates_from_noise_to_data() -> None:
    states, mask = _condition()
    model = ConstantVelocity(2.0)
    output = sample_rectified_flow(
        model,
        torch.ones(1, 1, 2, 2),
        positive_states=states,
        positive_mask=mask,
        negative_states=states,
        negative_mask=mask,
        steps=4,
        guidance_scale=1.0,
        solver="euler",
    )
    assert torch.allclose(output, torch.full_like(output, -1.0))
    assert model.calls == 4


def test_heun_uses_second_order_evaluations() -> None:
    states, mask = _condition()
    model = ConstantVelocity(1.0)
    output = sample_rectified_flow(
        model,
        torch.ones(1, 1, 1, 1),
        positive_states=states,
        positive_mask=mask,
        negative_states=states,
        negative_mask=mask,
        steps=3,
        solver="heun",
    )
    assert torch.allclose(output, torch.zeros_like(output))
    assert model.calls == 5
