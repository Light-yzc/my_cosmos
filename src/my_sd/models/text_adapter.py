from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(dtype=x.dtype) * weight


class TextAdapterAttention(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        if width % num_heads:
            raise ValueError("Text adapter width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.q_norm = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm = nn.Parameter(torch.ones(self.head_dim))

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = _rms_norm(q, self.q_norm).transpose(1, 2)
        k = _rms_norm(k, self.k_norm).transpose(1, 2)
        v = v.transpose(1, 2)

        sdpa_mask = None
        if attention_mask is not None:
            valid = attention_mask.to(device=x.device, dtype=torch.bool)
            sdpa_mask = valid[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        out = out.transpose(1, 2).reshape(batch, length, width)
        return self.out(out)


class TextAdapterBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(width * mlp_ratio)
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = TextAdapterAttention(width, num_heads)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width),
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        x = x + self.attention(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        if attention_mask is not None:
            x = x * attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        return x


class TextConditioningAdapter(nn.Module):
    """Maps a small text encoder to the 1024-wide Cosmos cross-attention context."""

    def __init__(
        self,
        input_dim: int,
        width: int = 1024,
        depth: int = 2,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            TextAdapterBlock(width, num_heads) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = self.input_projection(states)
        if attention_mask is not None:
            x = x * attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.output_norm(x)
