from __future__ import annotations

import unittest

from peek.cache import rebase_first_order


class RecordingSchedule:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def recover_noise(self, state, clean, timestep):
        self.calls.append(("recover", state, clean, timestep))
        return state - clean

    def add_noise(self, clean, noise, timestep):
        self.calls.append(("add", clean, noise, timestep))
        return clean + noise * timestep


class CacheTests(unittest.TestCase):
    def test_rebases_cached_state_with_recovered_noise(self) -> None:
        schedule = RecordingSchedule()

        result = rebase_first_order(
            schedule,
            previous_state=10.0,
            previous_clean=4.0,
            previous_timestep=3,
            target_timestep=2,
        )

        self.assertEqual(result, 16.0)
        self.assertEqual(
            schedule.calls,
            [("recover", 10.0, 4.0, 3), ("add", 4.0, 6.0, 2)],
        )

    def test_terminal_rebase_returns_clean_without_adding_noise(self) -> None:
        schedule = RecordingSchedule()

        result = rebase_first_order(
            schedule,
            previous_state=10.0,
            previous_clean=4.0,
            previous_timestep=3,
            target_timestep=None,
        )

        self.assertEqual(result, 4.0)
        self.assertEqual(schedule.calls, [("recover", 10.0, 4.0, 3)])


if __name__ == "__main__":
    unittest.main()
