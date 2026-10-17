"""Per-index random streams shared by baseline and PEEK generation."""

from __future__ import annotations

import torch


class StackedRandomGenerator:
    def __init__(self, device: torch.device, seeds: list[int]) -> None:
        self.device = device
        self.generators = [
            torch.Generator(device=device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size: tuple[int, ...], **kwargs: object) -> torch.Tensor:
        if size[0] != len(self.generators):
            raise ValueError("the leading tensor dimension must match the seed count")
        return torch.stack(
            [torch.randn(size[1:], generator=generator, **kwargs) for generator in self.generators]
        )

    def randint(self, high: int, size: tuple[int, ...], **kwargs: object) -> torch.Tensor:
        if size[0] != len(self.generators):
            raise ValueError("the leading tensor dimension must match the seed count")
        return torch.stack(
            [
                torch.randint(high, size=size[1:], generator=generator, **kwargs)
                for generator in self.generators
            ]
        )

    def one_hot_labels(
        self,
        label_dim: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        indices = self.randint(label_dim, size=(len(self.generators),), device=device)
        return torch.eye(label_dim, dtype=dtype, device=device)[indices]
