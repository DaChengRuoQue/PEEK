from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from peek.imagenet64.search import main, search_dpmpp_first_order, write_search_result


class ToyDenoiser:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state, sigma, class_labels=None):
        if class_labels is None:
            raise AssertionError("class labels were not forwarded")
        self.calls += 1
        return 0.05 * state + 0.001 * sigma


class Imagenet64SearchTests(unittest.TestCase):
    def test_search_combines_prefix_reuse_with_baseline_rebasing(self) -> None:
        net = ToyDenoiser()
        latents = torch.tensor([[[[0.01, -0.02]]]], dtype=torch.float64)
        labels = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

        result = search_dpmpp_first_order(
            net,
            latents,
            labels,
            metric=lambda states: abs(float(states[0, 0, 0, 0]) - 0.0004393391267333317),
            base_nfe=3,
            candidate_alphas=(1.0,),
        )

        self.assertEqual(result.alpha_schedule, (1.0, 0.0))
        self.assertEqual(result.actual_nfe, 4)
        self.assertEqual(tuple(nfe for nfe, _ in result.baseline_metrics), (4, 5))
        self.assertEqual(net.calls, 15)

    def test_writes_schedule_artifact_consumable_by_evaluation(self) -> None:
        net = ToyDenoiser()
        latents = torch.tensor([[[[0.01, -0.02]]]], dtype=torch.float64)
        labels = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        result = search_dpmpp_first_order(
            net,
            latents,
            labels,
            metric=lambda states: abs(float(states[0, 0, 0, 0]) - 0.0004393391267333317),
            base_nfe=3,
            candidate_alphas=(1.0,),
        )

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "schedule.json"
            write_search_result(destination, result, model="fixture-model")
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["model"], "fixture-model")
        self.assertEqual(payload["base_nfe"], 3)
        self.assertEqual(payload["matched_nfe"], 4)
        self.assertEqual(payload["alpha_schedule"], [1.0, 0.0])
        self.assertEqual(payload["peek_metric"], result.metric)
        self.assertEqual(
            payload["matched_baseline_metric"],
            dict(result.baseline_metrics)[result.actual_nfe],
        )
        self.assertEqual(payload["cache_modes"], ["selected_prefix", "baseline_rebase"])

    def test_dry_run_exposes_same_imagenet_model_and_both_search_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "--repo-root",
                        tmp,
                        "--search-images",
                        "128",
                        "--batch",
                        "128",
                        "--dry-run",
                    ]
                )
            payload = json.loads(output.getvalue())

        self.assertEqual(status, 0)
        self.assertEqual(payload["model"], "edm-imagenet-64x64-cond-adm")
        self.assertEqual(payload["base_nfe"], 6)
        self.assertEqual(payload["candidate_alphas"], [1.0])
        self.assertEqual(payload["cache_modes"], ["selected_prefix", "baseline_rebase"])
        self.assertTrue(payload["schedule_path"].endswith("greedy_alpha_schedule.json"))


if __name__ == "__main__":
    unittest.main()
