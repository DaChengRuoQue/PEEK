"""Cached greedy schedule search for the public ImageNet-64 evaluator."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from peek.search import (
    BaselineEvaluation,
    SearchEvaluation,
    SearchResult,
    greedy_schedule_search,
)

from .assets import load_asset_manifest
from .fid import FeatureMoments, calculate_fid
from .random import StackedRandomGenerator
from .sampler import (
    FirstOrderBaselineCache,
    advance_dpmpp_first_order_baseline,
    build_logsnr_schedule,
    sample_dpmpp_first_order_baseline,
    sample_dpmpp_first_order_suffix,
)


StateMetric = Callable[[torch.Tensor], float]
MODEL_NAME = "edm-imagenet-64x64-cond-adm"


class _BatchedDenoiser:
    def __init__(self, net, batch: int) -> None:
        self.net = net
        self.batch = int(batch)

    def __call__(self, states, sigma, class_labels=None):
        outputs = []
        for start in range(0, int(states.shape[0]), self.batch):
            labels = (
                class_labels[start : start + self.batch]
                if class_labels is not None
                else None
            )
            outputs.append(self.net(states[start : start + self.batch], sigma, labels))
        return torch.cat(outputs, dim=0)


def search_dpmpp_first_order(
    net,
    latents: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    metric: StateMetric,
    base_nfe: int = 6,
    candidate_alphas: Sequence[float] = (1.0,),
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> SearchResult:
    """Run greedy PEEK search with active prefix and baseline-state caches."""

    schedule = build_logsnr_schedule(
        int(base_nfe),
        float(sigma_min),
        float(sigma_max),
        latents.device,
        latents.dtype,
    )
    initial_prefix = latents * schedule[0]

    def evaluate_schedule(
        alpha_schedule,
        *,
        start_index: int,
        prefix_state: torch.Tensor | None,
        capture_next_prefix: bool,
    ) -> SearchEvaluation:
        source = initial_prefix if prefix_state is None else prefix_state
        final, next_prefix = sample_dpmpp_first_order_suffix(
            net,
            source,
            class_labels,
            schedule=schedule,
            alphas=tuple(float(alpha) for alpha in alpha_schedule),
            start_index=int(start_index),
            capture_next_prefix=bool(capture_next_prefix),
        )
        return SearchEvaluation(metric=float(metric(final)), next_prefix=next_prefix)

    def evaluate_baseline(nfe: int) -> BaselineEvaluation:
        final, cache = sample_dpmpp_first_order_baseline(
            net,
            latents,
            class_labels,
            num_evaluations=int(nfe),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
        )
        return BaselineEvaluation(metric=float(metric(final)), cache_state=cache)

    def advance_baseline(
        cache_state: FirstOrderBaselineCache,
        _source_nfe: int,
        target_nfe: int,
    ) -> BaselineEvaluation:
        cache = cache_state
        final = None
        while int(cache.num_evaluations) < int(target_nfe):
            final, cache = advance_dpmpp_first_order_baseline(
                net,
                cache,
                class_labels,
                target_num_evaluations=int(cache.num_evaluations) + 1,
                sigma_min=float(sigma_min),
                sigma_max=float(sigma_max),
            )
        if final is None:
            raise ValueError("target baseline budget must exceed the cached budget")
        return BaselineEvaluation(metric=float(metric(final)), cache_state=cache)

    def advance_prefix(prefix_state: torch.Tensor, step_index: int, alpha: float) -> torch.Tensor:
        local_alphas = [0.0] * (int(base_nfe) - 1)
        local_alphas[int(step_index)] = float(alpha)
        _final, next_prefix = sample_dpmpp_first_order_suffix(
            net,
            prefix_state,
            class_labels,
            schedule=schedule,
            alphas=tuple(local_alphas),
            start_index=int(step_index),
            capture_next_prefix=True,
            stop_after_prefix=True,
        )
        if next_prefix is None:
            raise RuntimeError("prefix advancement did not capture the next state")
        return next_prefix

    return greedy_schedule_search(
        base_nfe=int(base_nfe),
        candidate_alphas=candidate_alphas,
        evaluate_schedule=evaluate_schedule,
        evaluate_baseline=evaluate_baseline,
        advance_baseline=advance_baseline,
        initial_prefix=initial_prefix,
        advance_prefix=advance_prefix,
    )


def write_search_result(destination: Path, result: SearchResult, *, model: str) -> None:
    """Write a portable schedule artifact for subsequent evaluation."""

    path = destination.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_metrics = dict(result.baseline_metrics)
    matched_baseline_metric = baseline_metrics.get(int(result.actual_nfe))
    if matched_baseline_metric is None and int(result.selected_count) == 0:
        matched_baseline_metric = float(result.metric)
    if matched_baseline_metric is None:
        raise ValueError("search result is missing its final compute-matched baseline metric")
    payload = {
        "model": str(model),
        "solver": "first-order DPM-Solver++",
        "base_nfe": len(result.alpha_schedule) + 1,
        "matched_nfe": int(result.actual_nfe),
        "alpha_schedule": [float(alpha) for alpha in result.alpha_schedule],
        "selected_count": int(result.selected_count),
        "selected_metric": float(result.metric),
        "peek_metric": float(result.metric),
        "matched_baseline_metric": float(matched_baseline_metric),
        "baseline_metrics": [
            {"nfe": int(nfe), "metric": float(metric)}
            for nfe, metric in result.baseline_metrics
        ],
        "cache_modes": ["selected_prefix", "baseline_rebase"],
    }
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate alphas must be comma-separated numbers") from exc
    if not alphas:
        raise argparse.ArgumentTypeError("at least one candidate alpha is required")
    return alphas


def _run_config(
    repo_root: Path,
    *,
    search_images: int,
    batch: int,
    base_nfe: int,
    candidate_alphas: Sequence[float],
) -> dict:
    return {
        "repo_root": str(repo_root.resolve()),
        "model": MODEL_NAME,
        "solver": "first-order DPM-Solver++",
        "search_images": int(search_images),
        "batch": int(batch),
        "base_nfe": int(base_nfe),
        "candidate_alphas": [float(alpha) for alpha in candidate_alphas],
        "cache_modes": ["selected_prefix", "baseline_rebase"],
        "schedule_path": str(
            repo_root.resolve()
            / "runs"
            / "imagenet64_edm_schedule_search"
            / "greedy_alpha_schedule.json"
        ),
    }


def run_imagenet64_schedule_search(
    repo_root: Path,
    *,
    search_images: int,
    batch: int,
    base_nfe: int,
    candidate_alphas: Sequence[float],
    device: torch.device,
) -> SearchResult:
    """Calibrate an ImageNet schedule with FID and both search-time caches."""

    if int(search_images) < 2:
        raise ValueError("search_images must be at least 2")
    if int(batch) < 1:
        raise ValueError("batch must be positive")
    root = repo_root.resolve()
    assets = load_asset_manifest(root)
    with (root / assets.model.path).open("rb") as handle:
        net = pickle.load(handle)["ema"].to(device).eval().requires_grad_(False)
    with (root / assets.detector.path).open("rb") as handle:
        detector = pickle.load(handle).to(device).eval().requires_grad_(False)
    random = StackedRandomGenerator(device, list(range(int(search_images))))
    latents = random.randn(
        (int(search_images), int(net.img_channels), int(net.img_resolution), int(net.img_resolution)),
        dtype=torch.float32,
        device=device,
    )
    labels = random.one_hot_labels(int(net.label_dim), dtype=torch.float32, device=device)
    with np.load(root / assets.reference.path) as reference_data:
        reference = FeatureMoments(
            int(search_images),
            np.asarray(reference_data["mu"], dtype=np.float64),
            np.asarray(reference_data["sigma"], dtype=np.float64),
        )

    @torch.no_grad()
    def fid_metric(states: torch.Tensor) -> float:
        feature_batches: list[np.ndarray] = []
        for start in range(0, int(states.shape[0]), int(batch)):
            images = (
                (states[start : start + int(batch)] * 127.5 + 128.0)
                .clamp(0, 255)
                .to(torch.uint8)
            )
            features = detector(images, return_features=True)
            if isinstance(features, (tuple, list)):
                features = features[0]
            feature_batches.append(features.to(torch.float64).cpu().numpy())
        generated = FeatureMoments.from_features(np.concatenate(feature_batches, axis=0))
        return calculate_fid(generated, reference)

    result = search_dpmpp_first_order(
        _BatchedDenoiser(net, int(batch)),
        latents,
        labels,
        metric=fid_metric,
        base_nfe=int(base_nfe),
        candidate_alphas=candidate_alphas,
    )
    destination = (
        root
        / "runs"
        / "imagenet64_edm_schedule_search"
        / "greedy_alpha_schedule.json"
    )
    write_search_result(destination, result, model=MODEL_NAME)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--search-images", type=int, default=500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--base-nfe", type=int, default=6)
    parser.add_argument("--candidate-alphas", type=_parse_alphas, default=(1.0,))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = _run_config(
        args.repo_root,
        search_images=args.search_images,
        batch=args.batch,
        base_nfe=args.base_nfe,
        candidate_alphas=args.candidate_alphas,
    )
    if args.dry_run:
        print(json.dumps(config, indent=2), flush=True)
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ImageNet schedule search")
    result = run_imagenet64_schedule_search(
        args.repo_root,
        search_images=args.search_images,
        batch=args.batch,
        base_nfe=args.base_nfe,
        candidate_alphas=args.candidate_alphas,
        device=torch.device("cuda"),
    )
    config.update(
        {
            "alpha_schedule": [float(alpha) for alpha in result.alpha_schedule],
            "matched_nfe": int(result.actual_nfe),
            "selected_metric": float(result.metric),
        }
    )
    print(json.dumps(config, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
