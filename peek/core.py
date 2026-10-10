"""Framework-neutral PEEK and PEEK-G sampling primitives.

This module distills the correction loop from MultiStroke's tracked
``scripts/sample_celeba_restart.py`` implementation. Model and scheduler
details stay behind :class:`PeekBackend`, which also makes scheduler-state
rollback explicit for the provisional look-ahead step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, Sequence, TypeVar


StateT = TypeVar("StateT")
TimestepT = TypeVar("TimestepT")
SnapshotT = TypeVar("SnapshotT")
Method = Literal["peek", "peek_g"]


@dataclass(frozen=True)
class Prediction(Generic[StateT]):
    """A model prediction represented in clean-sample and direction forms."""

    clean: StateT
    direction: StateT


class PeekBackend(Protocol[StateT, TimestepT, SnapshotT]):
    """Operations needed to apply PEEK to a sampler backend."""

    def predict(self, state: StateT, timestep: TimestepT) -> Prediction[StateT]: ...

    def step(self, state: StateT, timestep: TimestepT, direction: StateT) -> StateT: ...

    def direction_from_clean(
        self, state: StateT, timestep: TimestepT, clean: StateT
    ) -> StateT: ...

    def snapshot(self) -> SnapshotT: ...

    def restore(self, snapshot: SnapshotT) -> None: ...


@dataclass(frozen=True)
class TraceEntry(Generic[StateT, TimestepT]):
    """One committed sampler step and the prediction selected for it."""

    step_index: int
    timestep: TimestepT
    alpha: float
    used_peek: bool
    current_clean: StateT
    lookahead_clean: StateT
    selected_clean: StateT
    current_direction: StateT
    lookahead_direction: StateT
    selected_direction: StateT
    state: StateT


@dataclass(frozen=True)
class SampleResult(Generic[StateT, TimestepT]):
    """Final state and a step-by-step correction trace."""

    final_state: StateT
    trace: tuple[TraceEntry[StateT, TimestepT], ...]


def interpolate(current: StateT, lookahead: StateT, alpha: float) -> StateT:
    """Interpolate between current and one-step-look-ahead predictions."""

    weight = float(alpha)
    return (1.0 - weight) * current + weight * lookahead  # type: ignore[operator, return-value]


def _validate(timesteps: Sequence[TimestepT], alpha_schedule: Sequence[float], method: str) -> None:
    if len(timesteps) == 0:
        raise ValueError("timesteps must contain at least one step")
    expected = len(timesteps) - 1
    if len(alpha_schedule) != expected:
        raise ValueError(
            f"alpha_schedule must contain {expected} values for {len(timesteps)} timesteps, "
            f"but received {len(alpha_schedule)}"
        )
    invalid = [float(alpha) for alpha in alpha_schedule if not 0.0 <= float(alpha) <= 1.0]
    if invalid:
        raise ValueError(f"alpha values must be in [0, 1], got {invalid}")
    if method not in {"peek", "peek_g"}:
        raise ValueError("method must be 'peek' or 'peek_g'")


def sample(
    initial_state: StateT,
    timesteps: Sequence[TimestepT],
    alpha_schedule: Sequence[float],
    backend: PeekBackend[StateT, TimestepT, SnapshotT],
    *,
    method: Method = "peek",
    record_trace: bool = False,
) -> SampleResult[StateT, TimestepT]:
    """Run an ordinary sampler augmented with a fixed PEEK schedule.

    A nonzero gate performs a provisional step, queries the model once at the
    following timestep, restores the backend, and commits the corrected step.
    ``peek`` interpolates clean predictions; ``peek_g`` interpolates prediction
    directions. Tracing is opt-in because retaining every intermediate tensor
    can be expensive for large batches.
    """

    _validate(timesteps, alpha_schedule, method)
    state = initial_state
    trace: list[TraceEntry[StateT, TimestepT]] = []

    for step_index, timestep in enumerate(timesteps):
        current = backend.predict(state, timestep)
        alpha = float(alpha_schedule[step_index]) if step_index < len(alpha_schedule) else 0.0
        used_peek = alpha != 0.0
        lookahead = current
        selected_clean = current.clean
        selected_direction = current.direction

        if used_peek:
            snapshot = backend.snapshot()
            try:
                provisional_state = backend.step(state, timestep, current.direction)
                lookahead = backend.predict(provisional_state, timesteps[step_index + 1])
            finally:
                backend.restore(snapshot)

            selected_clean = interpolate(current.clean, lookahead.clean, alpha)
            if method == "peek":
                selected_direction = backend.direction_from_clean(state, timestep, selected_clean)
            else:
                selected_direction = interpolate(current.direction, lookahead.direction, alpha)

        state = backend.step(state, timestep, selected_direction)
        if record_trace:
            trace.append(
                TraceEntry(
                    step_index=step_index,
                    timestep=timestep,
                    alpha=alpha,
                    used_peek=used_peek,
                    current_clean=current.clean,
                    lookahead_clean=lookahead.clean,
                    selected_clean=selected_clean,
                    current_direction=current.direction,
                    lookahead_direction=lookahead.direction,
                    selected_direction=selected_direction,
                    state=state,
                )
            )

    return SampleResult(final_state=state, trace=tuple(trace))
