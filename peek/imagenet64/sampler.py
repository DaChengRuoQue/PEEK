"""First-order DPM-Solver++ with the fixed PEEK correction path."""

from __future__ import annotations

from collections.abc import Callable

import torch

from peek.core import interpolate


Denoiser = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def build_logsnr_schedule(
    num_evaluations: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_evaluations < 1:
        raise ValueError("num_evaluations must be positive")
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max")
    start = torch.log(torch.tensor(float(sigma_max), device=device, dtype=dtype))
    end = torch.log(torch.tensor(float(sigma_min), device=device, dtype=dtype))
    return torch.linspace(start, end, num_evaluations + 1, device=device, dtype=dtype).exp()


def _dynamic_threshold(clean: torch.Tensor) -> torch.Tensor:
    scale = torch.quantile(clean.abs().reshape(clean.shape[0], -1), 0.995, dim=1)
    scale = torch.maximum(scale, torch.ones_like(scale))
    scale = scale.reshape((clean.shape[0],) + (1,) * (clean.ndim - 1))
    return clean.clamp(-scale, scale) / scale


def _first_order_update(
    state: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    ratio = next_sigma / sigma
    return ratio * state + (1.0 - ratio) * clean


@torch.no_grad()
def sample_dpmpp_first_order(
    net: Denoiser,
    latents: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    num_evaluations: int,
    gates: tuple[int, ...],
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    if gates and len(gates) != num_evaluations - 1:
        raise ValueError(
            f"expected {num_evaluations - 1} gates for {num_evaluations} base evaluations, "
            f"received {len(gates)}"
        )
    if any(gate not in (0, 1) for gate in gates):
        raise ValueError("PEEK gates must be binary")

    schedule = build_logsnr_schedule(
        num_evaluations,
        sigma_min,
        sigma_max,
        latents.device,
        latents.dtype,
    )
    state = latents * schedule[0]
    for step_index, (sigma, next_sigma) in enumerate(zip(schedule[:-1], schedule[1:])):
        current_clean = _dynamic_threshold(net(state, sigma, class_labels))
        selected_clean = current_clean
        provisional = _first_order_update(state, sigma, next_sigma, current_clean)
        if step_index < len(gates) and gates[step_index]:
            lookahead_clean = _dynamic_threshold(net(provisional, next_sigma, class_labels))
            selected_clean = interpolate(current_clean, lookahead_clean, 1.0)
        state = _first_order_update(state, sigma, next_sigma, selected_clean)
    return state
