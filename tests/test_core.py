from __future__ import annotations

import unittest

from peek.core import Prediction, interpolate, sample


class ScalarBackend:
    def __init__(self) -> None:
        self.cursor = 0
        self.predict_calls = 0

    def predict(self, state: float, timestep: int) -> Prediction[float]:
        self.predict_calls += 1
        return Prediction(clean=state + timestep, direction=float(timestep))

    def step(self, state: float, timestep: int, direction: float) -> float:
        self.cursor += 1
        return state - direction

    def direction_from_clean(self, state: float, timestep: int, clean: float) -> float:
        return clean - state

    def snapshot(self) -> int:
        return self.cursor

    def restore(self, snapshot: int) -> None:
        self.cursor = snapshot


class AmbiguousTruthSequence:
    def __init__(self, values) -> None:
        self.values = list(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __bool__(self) -> bool:
        raise RuntimeError("sequence truth value is ambiguous")


class CoreTests(unittest.TestCase):
    def test_interpolate_uses_requested_weight(self) -> None:
        self.assertEqual(interpolate(2.0, 10.0, 0.25), 4.0)

    def test_zero_schedule_matches_ordinary_sampling(self) -> None:
        backend = ScalarBackend()

        result = sample(10.0, [3, 2], [0.0], backend, record_trace=True)

        self.assertEqual(result.final_state, 5.0)
        self.assertEqual(backend.cursor, 2)
        self.assertEqual(len(result.trace), 2)
        self.assertFalse(result.trace[0].used_peek)

    def test_peek_interpolates_clean_prediction(self) -> None:
        backend = ScalarBackend()

        result = sample(10.0, [3, 2], [1.0], backend, method="peek", record_trace=True)

        self.assertEqual(result.final_state, 9.0)
        self.assertEqual(result.trace[0].current_clean, 13.0)
        self.assertEqual(result.trace[0].lookahead_clean, 9.0)
        self.assertEqual(result.trace[0].selected_clean, 9.0)

    def test_peek_g_interpolates_prediction_direction(self) -> None:
        backend = ScalarBackend()

        result = sample(10.0, [3, 2], [1.0], backend, method="peek_g", record_trace=True)

        self.assertEqual(result.final_state, 6.0)
        self.assertEqual(result.trace[0].selected_direction, 2.0)

    def test_lookahead_probe_restores_backend_state(self) -> None:
        backend = ScalarBackend()

        sample(10.0, [3, 2], [1.0], backend)

        self.assertEqual(backend.cursor, 2)

    def test_trace_is_opt_in_to_avoid_retaining_model_tensors(self) -> None:
        backend = ScalarBackend()

        result = sample(10.0, [3, 2], [1.0], backend)

        self.assertEqual(result.trace, ())

    def test_tensor_like_timestep_sequence_does_not_require_truth_testing(self) -> None:
        backend = ScalarBackend()

        result = sample(10.0, AmbiguousTruthSequence([3, 2]), [0.0], backend)

        self.assertEqual(result.final_state, 5.0)

    def test_invalid_schedule_fails_before_backend_prediction(self) -> None:
        cases = [
            ([], [], "peek"),
            ([3, 2], [], "peek"),
            ([3, 2], [-0.1], "peek"),
            ([3, 2], [1.1], "peek"),
            ([3, 2], [0.0], "unknown"),
        ]
        for timesteps, alphas, method in cases:
            with self.subTest(timesteps=timesteps, alphas=alphas, method=method):
                backend = ScalarBackend()
                with self.assertRaises(ValueError):
                    sample(10.0, timesteps, alphas, backend, method=method)
                self.assertEqual(backend.predict_calls, 0)


if __name__ == "__main__":
    unittest.main()
