from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .text_adapter import TextConditioningAdapter


def _rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(dtype=x.dtype) * weight


def _rotate_pairs(x: Tensor) -> Tensor:
    paired = x.unflatten(-1, (-1, 2))
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


@dataclass(slots=True)
class CosmosDiTConfig:
    latent_channels: int = 48
    patch_size: int = 2
    hidden_size: int = 1280
    depth: int = 27
    num_heads: int = 20
    mlp_ratio: float = 4.0
    context_dim: int = 1024
    text_input_dim: int = 640
    text_adapter_depth: int = 2
    text_adapter_heads: int = 16
    adaln_rank: int = 256
    rope_theta: float = 10000.0
    rope_extrapolation: float = 4.0
    gradient_checkpointing: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CosmosDiTConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown CosmosDiT config keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 4:
            raise ValueError("Attention head dimension must be divisible by four for 2D RoPE")
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive")
        if self.depth < 1:
            raise ValueError("depth must be positive")


class AxialRoPE2D(nn.Module):
    """Learnable-frequency axial RoPE for a row-major image token grid."""

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        extrapolation: float = 4.0,
    ) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by four")
        if extrapolation <= 0:
            raise ValueError("extrapolation must be positive")
        pairs_per_axis = head_dim // 4
        indices = torch.arange(pairs_per_axis, dtype=torch.float32)
        inv_freq = theta ** (-indices / max(pairs_per_axis, 1))
        self.log_inv_freq = nn.Parameter(inv_freq.log())
        self.extrapolation = float(extrapolation)

    def forward(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        inv_freq = self.log_inv_freq.exp().to(device=device)
        y_phase = y.flatten()[:, None] * inv_freq[None, :] / self.extrapolation
        x_phase = x.flatten()[:, None] * inv_freq[None, :] / self.extrapolation
        phase = torch.cat(
            (y_phase.repeat_interleave(2, dim=-1), x_phase.repeat_interleave(2, dim=-1)),
            dim=-1,
        )
        return phase.cos().to(dtype=dtype), phase.sin().to(dtype=dtype)


class SelfAttention2D(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.q_norm = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm = nn.Parameter(torch.ones(self.head_dim))

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = _rms_norm(q, self.q_norm)
        k = _rms_norm(k, self.k_norm)
        cos, sin = rope
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q = q * cos + _rotate_pairs(q) * sin
        k = k * cos + _rotate_pairs(k) * sin
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, length, width)
        return self.output(out)


class CrossAttention(nn.Module):
    def __init__(self, width: int, context_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(context_dim, width * 2)
        self.output = nn.Linear(width, width)
        self.q_norm = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm = nn.Parameter(torch.ones(self.head_dim))

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Tensor | None,
    ) -> Tensor:
        batch, query_length, width = x.shape
        context_length = context.shape[1]
        q = self.query(x).view(batch, query_length, self.num_heads, self.head_dim)
        kv = self.key_value(context).view(
            batch, context_length, 2, self.num_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        q = _rms_norm(q, self.q_norm).transpose(1, 2)
        k = _rms_norm(k, self.k_norm).transpose(1, 2)
        v = v.transpose(1, 2)

        bias = None
        if context_mask is not None:
            valid = context_mask.to(device=x.device, dtype=torch.bool)
            bias = torch.zeros(
                (batch, 1, 1, context_length), device=x.device, dtype=q.dtype
            )
            bias.masked_fill_(~valid[:, None, None, :], torch.finfo(q.dtype).min)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(batch, query_length, width)
        return self.output(out)


class FeedForward(nn.Module):
    def __init__(self, width: int, ratio: float) -> None:
        super().__init__()
        hidden = int(width * ratio)
        self.layers = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class AdaLNLoRA(nn.Module):
    """A shared AdaLN modulation plus a block-local rank-reduced residual."""

    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width * 3, bias=False)

    def forward(self, conditioning: Tensor, shared: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        local = self.up(self.down(F.silu(conditioning)))
        shift, scale, gate = (shared + local).chunk(3, dim=-1)
        return shift[:, None, :], scale[:, None, :], gate[:, None, :]


class FinalAdaLNLoRA(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.width = width
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width * 2, bias=False)

    def forward(self, conditioning: Tensor, shared: Tensor) -> tuple[Tensor, Tensor]:
        local = self.up(self.down(F.silu(conditioning)))
        return (shared[:, : self.width * 2] + local).chunk(2, dim=-1)


class CosmosBlock(nn.Module):
    def __init__(
        self,
        width: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float,
        adaln_rank: int,
    ) -> None:
        super().__init__()
        norm = partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6)
        self.self_norm = norm(width)
        self.cross_norm = norm(width)
        self.mlp_norm = norm(width)
        self.self_attention = SelfAttention2D(width, num_heads)
        self.cross_attention = CrossAttention(width, context_dim, num_heads)
        self.feed_forward = FeedForward(width, mlp_ratio)
        self.self_modulation = AdaLNLoRA(width, adaln_rank)
        self.cross_modulation = AdaLNLoRA(width, adaln_rank)
        self.mlp_modulation = AdaLNLoRA(width, adaln_rank)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale) + shift

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Tensor | None,
        conditioning: Tensor,
        shared_modulation: Tensor,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        shift, scale, gate = self.self_modulation(conditioning, shared_modulation)
        residual = self._modulate(self.self_norm(x), shift, scale)
        x = x + gate * self.self_attention(residual, rope)

        shift, scale, gate = self.cross_modulation(conditioning, shared_modulation)
        residual = self._modulate(self.cross_norm(x), shift, scale)
        x = x + gate * self.cross_attention(residual, context, context_mask)

        shift, scale, gate = self.mlp_modulation(conditioning, shared_modulation)
        residual = self._modulate(self.mlp_norm(x), shift, scale)
        return x + gate * self.feed_forward(residual)


def sinusoidal_timestep_embedding(
    timesteps: Tensor,
    width: int,
    max_period: float = 10000.0,
) -> Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
    if width % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.projection = nn.Linear(width, width, bias=False)
        self.shared_modulation = nn.Linear(width, width * 3, bias=False)

    def forward(self, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        conditioning = sinusoidal_timestep_embedding(
            timesteps * 1000.0, self.width
        ).to(dtype=self.projection.weight.dtype)
        shared = self.shared_modulation(F.silu(self.projection(conditioning)))
        return conditioning, shared


class CosmosDiT(nn.Module):
    """
    Static-image adaptation of Cosmos-Predict2 MiniTrainDIT.

    This intentionally is not checkpoint-compatible with NVIDIA's released model:
    it consumes Wan2.2 f16c48 latents and T5Gemma-2 encoder states.
    """

    def __init__(self, config: CosmosDiTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.hidden_size
        head_dim = width // config.num_heads

        self.patch_embed = nn.Conv2d(
            config.latent_channels,
            width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.text_adapter = TextConditioningAdapter(
            input_dim=config.text_input_dim,
            width=config.context_dim,
            depth=config.text_adapter_depth,
            num_heads=config.text_adapter_heads,
        )
        self.timestep = TimestepEmbedder(width)
        self.rope = AxialRoPE2D(
            head_dim,
            theta=config.rope_theta,
            extrapolation=config.rope_extrapolation,
        )
        self.blocks = nn.ModuleList(
            CosmosBlock(
                width=width,
                context_dim=config.context_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                adaln_rank=config.adaln_rank,
            )
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = FinalAdaLNLoRA(width, config.adaln_rank)
        self.output_projection = nn.Linear(
            width,
            config.patch_size * config.patch_size * config.latent_channels,
            bias=False,
        )
        self.apply(self._initialize_module)
        self._zero_initialize_conditioning()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _zero_initialize_conditioning(self) -> None:
        for module in self.modules():
            if isinstance(module, (AdaLNLoRA, FinalAdaLNLoRA)):
                nn.init.zeros_(module.up.weight)
        nn.init.zeros_(self.output_projection.weight)

    def _run_block(
        self,
        block: CosmosBlock,
        x: Tensor,
        context: Tensor,
        context_mask: Tensor | None,
        conditioning: Tensor,
        shared_modulation: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        return block(
            x,
            context,
            context_mask,
            conditioning,
            shared_modulation,
            (rope_cos, rope_sin),
        )

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        text_states: Tensor,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        if latents.ndim != 4:
            raise ValueError("latents must have shape [batch, channels, height, width]")
        if latents.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"Expected {self.config.latent_channels} latent channels, got {latents.shape[1]}"
            )
        patch = self.config.patch_size
        if latents.shape[-2] % patch or latents.shape[-1] % patch:
            raise ValueError("Latent height and width must be divisible by patch_size")
        if timesteps.ndim != 1 or timesteps.shape[0] != latents.shape[0]:
            raise ValueError("timesteps must have shape [batch]")

        context = self.text_adapter(text_states, text_mask)
        conditioning, shared = self.timestep(timesteps)
        spatial = self.patch_embed(latents)
        grid_height, grid_width = spatial.shape[-2:]
        x = spatial.flatten(2).transpose(1, 2)
        rope_cos, rope_sin = self.rope(
            grid_height,
            grid_width,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(
                    self._run_block,
                    block,
                    x,
                    context,
                    text_mask,
                    conditioning,
                    shared,
                    rope_cos,
                    rope_sin,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    context,
                    text_mask,
                    conditioning,
                    shared,
                    (rope_cos, rope_sin),
                )

        shift, scale = self.final_modulation(conditioning, shared)
        x = self.final_norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        patches = self.output_projection(x)
        batch = latents.shape[0]
        channels = self.config.latent_channels
        output = patches.view(
            batch, grid_height, grid_width, patch, patch, channels
        )
        return output.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, channels, grid_height * patch, grid_width * patch
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
