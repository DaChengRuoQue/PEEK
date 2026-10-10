"""Dependency-free smoke example for the PEEK correction loop."""

from __future__ import annotations

import json

from peek import Prediction, sample


class ToyBackend:
    """Small deterministic backend that makes the look-ahead path observable."""

    def __init__(self) -> None:
        self.step_count = 0

    def predict(self, state: float, timestep: int) -> Prediction[float]:
        return Prediction(clean=state + timestep, direction=float(timestep))

    def step(self, state: float, timestep: int, direction: float) -> float:
        self.step_count += 1
        return state - direction

    def direction_from_clean(self, state: float, timestep: int, clean: float) -> float:
        return clean - state

    def snapshot(self) -> int:
        return self.step_count

    def restore(self, snapshot: int) -> None:
        self.step_count = snapshot


def main() -> None:
    timesteps = [3, 2]
    ordinary = sample(10.0, timesteps, [0.0], ToyBackend())
    peek = sample(10.0, timesteps, [1.0], ToyBackend(), record_trace=True)
    print(
        json.dumps(
            {
                "ordinary_final_state": ordinary.final_state,
                "peek_final_state": peek.final_state,
                "peek_used_at_steps": [entry.step_index for entry in peek.trace if entry.used_peek],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
