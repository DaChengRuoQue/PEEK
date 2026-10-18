"""First-order DPM-Solver++ with the fixed PEEK correction path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from peek.cache import rebase_first_order
from peek.core import interpolate


Denoiser = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


@dataclass(frozen=True)
class FirstOrderBaselineCache:
    """Pre-final state needed to extend an ordinary first-order baseline."""

    num_evaluations: int
    previous_state: torch.Tensor
    previous_clean: torch.Tensor
    previous_sigma: torch.Tensor


class _VENoiseSchedule:
    def recover_noise(
        self,
        state: torch.Tensor,
        clean: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return (state - clean) / sigma

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return clean + sigma * noise


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
def sample_dpmpp_first_order_suffix(
    net: Denoiser,
    state: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    schedule: torch.Tensor,
    alphas: tuple[float, ...],
    start_index: int = 0,
    capture_next_prefix: bool = False,
    stop_after_prefix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_evaluations = int(schedule.numel()) - 1
    if len(alphas) != num_evaluations - 1:
        raise ValueError(
            f"expected {num_evaluations - 1} alphas for {num_evaluations} base evaluations, "
            f"received {len(alphas)}"
        )
    if any(not 0.0 <= float(alpha) <= 1.0 for alpha in alphas):
        raise ValueError("PEEK alphas must be in [0, 1]")
    if not 0 <= int(start_index) < num_evaluations:
        raise ValueError("start_index must select a base evaluation")

    next_prefix = None
    for step_index in range(int(start_index), num_evaluations):
        sigma = schedule[step_index]
        next_sigma = schedule[step_index + 1]
        current_clean = _dynamic_threshold(net(state, sigma, class_labels))
        selected_clean = current_clean
        provisional = _first_order_update(state, sigma, next_sigma, current_clean)
        alpha = float(alphas[step_index]) if step_index < len(alphas) else 0.0
        if alpha != 0.0:
            lookahead_clean = _dynamic_threshold(net(provisional, next_sigma, class_labels))
            selected_clean = interpolate(current_clean, lookahead_clean, alpha)
        state = _first_order_update(state, sigma, next_sigma, selected_clean)
        if capture_next_prefix and step_index == int(start_index):
            next_prefix = state.detach().clone()
            if stop_after_prefix:
                break
    return state, next_prefix


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
    alphas = tuple(float(gate) for gate in gates) if gates else (0.0,) * (num_evaluations - 1)
    schedule = build_logsnr_schedule(
        num_evaluations,
        sigma_min,
        sigma_max,
        latents.device,
        latents.dtype,
    )
    final, _ = sample_dpmpp_first_order_suffix(
        net,
        latents * schedule[0],
        class_labels,
        schedule=schedule,
        alphas=alphas,
    )
    return final


@torch.no_grad()
def sample_dpmpp_first_order_baseline(
    net: Denoiser,
    latents: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    num_evaluations: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> tuple[torch.Tensor, FirstOrderBaselineCache]:
    schedule = build_logsnr_schedule(
        num_evaluations,
        sigma_min,
        sigma_max,
        latents.device,
        latents.dtype,
    )
    state = latents * schedule[0]
    previous_state = None
    previous_clean = None
    previous_sigma = None
    for step_index, (sigma, next_sigma) in enumerate(zip(schedule[:-1], schedule[1:])):
        clean = _dynamic_threshold(net(state, sigma, class_labels))
        if step_index == num_evaluations - 1:
            previous_state = state.detach().clone()
            previous_clean = clean.detach().clone()
            previous_sigma = sigma.detach().clone()
        state = _first_order_update(state, sigma, next_sigma, clean)
    assert previous_state is not None and previous_clean is not None and previous_sigma is not None
    return state, FirstOrderBaselineCache(
        num_evaluations=int(num_evaluations),
        previous_state=previous_state,
        previous_clean=previous_clean,
        previous_sigma=previous_sigma,
    )


@torch.no_grad()
def advance_dpmpp_first_order_baseline(
    net: Denoiser,
    cache: FirstOrderBaselineCache,
    class_labels: torch.Tensor | None,
    *,
    target_num_evaluations: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> tuple[torch.Tensor, FirstOrderBaselineCache]:
    if int(target_num_evaluations) <= int(cache.num_evaluations):
        raise ValueError("target_num_evaluations must increase the cached baseline budget")
    schedule = build_logsnr_schedule(
        int(target_num_evaluations),
        sigma_min,
        sigma_max,
        cache.previous_state.device,
        cache.previous_state.dtype,
    )
    target_sigma = schedule[-2]
    final_sigma = schedule[-1]
    rebased = rebase_first_order(
        _VENoiseSchedule(),
        previous_state=cache.previous_state,
        previous_clean=cache.previous_clean,
        previous_timestep=cache.previous_sigma,
        target_timestep=target_sigma,
    )
    clean = _dynamic_threshold(net(rebased, target_sigma, class_labels))
    final = _first_order_update(rebased, target_sigma, final_sigma, clean)
    return final, FirstOrderBaselineCache(
        num_evaluations=int(target_num_evaluations),
        previous_state=rebased.detach().clone(),
        previous_clean=clean.detach().clone(),
        previous_sigma=target_sigma.detach().clone(),
    )
