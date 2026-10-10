"""Cache primitives used by PEEK's greedy schedule search."""

from __future__ import annotations

from typing import Protocol, TypeVar


StateT = TypeVar("StateT")
NoiseT = TypeVar("NoiseT")
TimestepT = TypeVar("TimestepT")


class NoiseSchedule(Protocol[StateT, NoiseT, TimestepT]):
    """Adapter for moving a cached first-order state between timesteps."""

    def recover_noise(
        self, state: StateT, clean: StateT, timestep: TimestepT
    ) -> NoiseT: ...

    def add_noise(
        self, clean: StateT, noise: NoiseT, timestep: TimestepT
    ) -> StateT: ...


def rebase_first_order(
    schedule: NoiseSchedule[StateT, NoiseT, TimestepT],
    *,
    previous_state: StateT,
    previous_clean: StateT,
    previous_timestep: TimestepT,
    target_timestep: TimestepT | None,
) -> StateT:
    """Move a cached state to another noise level while preserving its noise.

    ``None`` denotes the terminal clean state. The operation mirrors the
    baseline noise-schedule rebasing cache used by the source PEEK search.
    """

    noise = schedule.recover_noise(previous_state, previous_clean, previous_timestep)
    if target_timestep is None:
        return previous_clean
    return schedule.add_noise(previous_clean, noise, target_timestep)
