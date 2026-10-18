from __future__ import annotations

import unittest

import torch

from peek.imagenet64.sampler import (
    advance_dpmpp_first_order_baseline,
    build_logsnr_schedule,
    sample_dpmpp_first_order,
    sample_dpmpp_first_order_baseline,
    sample_dpmpp_first_order_suffix,
)


class ToyDenoiser:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_labels: list[torch.Tensor] = []

    def __call__(
        self,
        state: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        if class_labels is None:
            raise AssertionError("class labels were not forwarded")
        self.seen_labels.append(class_labels.clone())
        return 0.05 * state + 0.001 * sigma


class Imagenet64SamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.latents = torch.tensor([[[[0.01, -0.02]]]], dtype=torch.float64)
        self.labels = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

    def test_logsnr_schedule_has_one_interval_per_base_evaluation(self) -> None:
        schedule = build_logsnr_schedule(
            6,
            sigma_min=0.002,
            sigma_max=80.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        self.assertEqual(schedule.shape, (7,))
        torch.testing.assert_close(
            schedule,
            torch.tensor(
                [
                    80.0,
                    13.679807573413573,
                    2.3392141905702926,
                    0.4,
                    0.06839903786706789,
                    0.011696070952851457,
                    0.002,
                ],
                dtype=torch.float64,
            ),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_nine_evaluation_baseline_matches_fixed_literal(self) -> None:
        net = ToyDenoiser()

        result = sample_dpmpp_first_order(
            net,
            self.latents,
            self.labels,
            num_evaluations=9,
            gates=(),
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[[[0.00011636426272891822, -0.00004]]]], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-14,
        )
        self.assertEqual(net.calls, 9)
        self.assertTrue(all(torch.equal(labels, self.labels) for labels in net.seen_labels))

    def test_six_evaluation_peek_matches_fixed_literal_at_nine_calls(self) -> None:
        net = ToyDenoiser()

        result = sample_dpmpp_first_order(
            net,
            self.latents,
            self.labels,
            num_evaluations=6,
            gates=(1, 1, 1, 0, 0),
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[[[0.00009377113858853572, -0.00004]]]], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-14,
        )
        self.assertEqual(net.calls, 9)

    def test_sampler_rejects_gate_schedule_with_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "5 gates"):
            sample_dpmpp_first_order(
                ToyDenoiser(),
                self.latents,
                self.labels,
                num_evaluations=6,
                gates=(1, 1, 1),
            )

    def test_cached_prefix_reproduces_the_selected_full_trajectory_suffix(self) -> None:
        schedule = build_logsnr_schedule(
            3,
            sigma_min=0.002,
            sigma_max=80.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        initial = self.latents * schedule[0]
        full_net = ToyDenoiser()

        full, prefix = sample_dpmpp_first_order_suffix(
            full_net,
            initial,
            self.labels,
            schedule=schedule,
            alphas=(1.0, 1.0),
            start_index=0,
            capture_next_prefix=True,
        )
        resumed_net = ToyDenoiser()
        resumed, _ = sample_dpmpp_first_order_suffix(
            resumed_net,
            prefix,
            self.labels,
            schedule=schedule,
            alphas=(1.0, 1.0),
            start_index=1,
        )

        torch.testing.assert_close(resumed, full, rtol=1e-12, atol=1e-14)
        self.assertEqual(full_net.calls, 5)
        self.assertEqual(resumed_net.calls, 3)

    def test_prefix_only_mode_stops_after_one_base_update(self) -> None:
        schedule = build_logsnr_schedule(
            4,
            sigma_min=0.002,
            sigma_max=80.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        net = ToyDenoiser()

        stopped, prefix = sample_dpmpp_first_order_suffix(
            net,
            self.latents * schedule[0],
            self.labels,
            schedule=schedule,
            alphas=(0.0, 0.0, 0.0),
            start_index=0,
            capture_next_prefix=True,
            stop_after_prefix=True,
        )

        torch.testing.assert_close(stopped, prefix, rtol=0, atol=0)
        self.assertEqual(net.calls, 1)

    def test_rebased_baseline_uses_one_new_denoiser_evaluation(self) -> None:
        exact_net = ToyDenoiser()
        _final, cache = sample_dpmpp_first_order_baseline(
            exact_net,
            self.latents,
            self.labels,
            num_evaluations=3,
        )
        rebase_net = ToyDenoiser()

        rebased_final, next_cache = advance_dpmpp_first_order_baseline(
            rebase_net,
            cache,
            self.labels,
            target_num_evaluations=4,
        )

        torch.testing.assert_close(
            rebased_final,
            torch.tensor([[[[0.000713375653172604, -0.00004]]]], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-14,
        )
        self.assertEqual(exact_net.calls, 3)
        self.assertEqual(rebase_net.calls, 1)
        self.assertEqual(next_cache.num_evaluations, 4)


if __name__ == "__main__":
    unittest.main()
