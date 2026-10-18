from __future__ import annotations

import unittest

import peek.search as search
from peek.search import SearchEvaluation, greedy_schedule_search


class SearchTests(unittest.TestCase):
    def test_accepts_only_candidates_better_than_current_and_matched_baseline(self) -> None:
        scores = {
            (0.0, 0.0): 10.0,
            (0.5, 0.0): 8.0,
            (1.0, 0.0): 7.0,
            (1.0, 0.5): 6.0,
            (1.0, 1.0): 6.2,
        }
        baseline_calls: list[int] = []

        def evaluate(schedule, **_kwargs):
            return scores[tuple(schedule)]

        def baseline(nfe: int) -> float:
            baseline_calls.append(nfe)
            return {4: 7.5, 5: 5.5}[nfe]

        result = greedy_schedule_search(
            base_nfe=3,
            candidate_alphas=[0.5, 1.0],
            evaluate_schedule=evaluate,
            evaluate_baseline=baseline,
        )

        self.assertEqual(result.alpha_schedule, (1.0, 0.0))
        self.assertEqual(result.selected_count, 1)
        self.assertEqual(result.actual_nfe, 4)
        self.assertEqual(baseline_calls, [4, 5])
        self.assertTrue(result.records[0].accepted)
        self.assertFalse(result.records[1].accepted)
        self.assertTrue(result.records[1].short_circuited)

    def test_memoizes_repeated_compute_matched_baseline(self) -> None:
        baseline_calls: list[int] = []

        def baseline(nfe: int) -> float:
            baseline_calls.append(nfe)
            return 0.0

        result = greedy_schedule_search(
            base_nfe=3,
            candidate_alphas=[1.0],
            evaluate_schedule=lambda _schedule, **_kwargs: 10.0,
            evaluate_baseline=baseline,
        )

        self.assertEqual(result.alpha_schedule, (0.0, 0.0))
        self.assertEqual(baseline_calls, [4])

    def test_advances_cached_baseline_when_matched_budget_increases(self) -> None:
        exact_calls: list[int] = []
        advance_calls: list[tuple[object, int, int]] = []

        def evaluate(schedule, **_kwargs):
            return 10.0 - 2.0 * sum(schedule)

        def baseline(nfe: int):
            exact_calls.append(nfe)
            return search.BaselineEvaluation(metric=9.0, cache_state="baseline-4")

        def advance(cache_state, source_nfe: int, target_nfe: int):
            advance_calls.append((cache_state, source_nfe, target_nfe))
            return search.BaselineEvaluation(metric=7.0, cache_state="baseline-5")

        result = greedy_schedule_search(
            base_nfe=3,
            candidate_alphas=[1.0],
            evaluate_schedule=evaluate,
            evaluate_baseline=baseline,
            advance_baseline=advance,
        )

        self.assertEqual(result.alpha_schedule, (1.0, 1.0))
        self.assertEqual(exact_calls, [4])
        self.assertEqual(advance_calls, [("baseline-4", 4, 5)])
        self.assertEqual(result.baseline_metrics, ((4, 9.0), (5, 7.0)))

    def test_prefix_cache_reuses_selected_candidate_state(self) -> None:
        calls: list[tuple[int, object, bool]] = []

        def evaluate(schedule, *, start_index, prefix_state, capture_next_prefix):
            calls.append((start_index, prefix_state, capture_next_prefix))
            metric = 10.0 - sum(schedule) * 2.0
            next_prefix = f"p{start_index + 1}" if capture_next_prefix else None
            return SearchEvaluation(metric=metric, next_prefix=next_prefix)

        result = greedy_schedule_search(
            base_nfe=3,
            candidate_alphas=[1.0],
            evaluate_schedule=evaluate,
            evaluate_baseline=lambda _nfe: 9.0,
            initial_prefix="p0",
            advance_prefix=lambda *_args: self.fail("selected prefix should be reused"),
        )

        self.assertEqual(result.alpha_schedule, (1.0, 1.0))
        self.assertEqual(calls, [(0, None, False), (0, "p0", True), (1, "p1", False)])

    def test_rejected_candidate_advances_zero_gate_prefix(self) -> None:
        advances: list[tuple[object, int, float]] = []

        def advance(prefix, step_index, alpha):
            advances.append((prefix, step_index, alpha))
            return "advanced"

        greedy_schedule_search(
            base_nfe=3,
            candidate_alphas=[1.0],
            evaluate_schedule=lambda _schedule, **_kwargs: 10.0,
            evaluate_baseline=lambda _nfe: 0.0,
            initial_prefix="p0",
            advance_prefix=advance,
        )

        self.assertEqual(advances, [("p0", 0, 0.0)])

    def test_missing_prefix_advancer_fails_before_expensive_evaluation(self) -> None:
        calls = []

        with self.assertRaises(ValueError):
            greedy_schedule_search(
                base_nfe=3,
                candidate_alphas=[1.0],
                evaluate_schedule=lambda *_args, **_kwargs: calls.append(True),
                evaluate_baseline=lambda _nfe: 0.0,
                initial_prefix="p0",
            )

        self.assertEqual(calls, [])

    def test_rejects_invalid_search_configuration_before_evaluation(self) -> None:
        invalid_cases = [
            (1, [1.0]),
            (3, []),
            (3, [0.0]),
            (3, [1.1]),
            (3, [1.0, 0.5]),
            (3, [0.5, 0.5]),
        ]
        for base_nfe, candidates in invalid_cases:
            with self.subTest(base_nfe=base_nfe, candidates=candidates):
                calls = []
                with self.assertRaises(ValueError):
                    greedy_schedule_search(
                        base_nfe=base_nfe,
                        candidate_alphas=candidates,
                        evaluate_schedule=lambda *_args, **_kwargs: calls.append(True),
                        evaluate_baseline=lambda _nfe: 0.0,
                    )
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
