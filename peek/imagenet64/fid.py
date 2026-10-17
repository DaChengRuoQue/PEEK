"""Distributed NVIDIA-compatible FID for repository-local ImageNet samples."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from PIL import Image

from .assets import load_asset_manifest, sha256_file
from .generate import GenerationConfig


@dataclass(frozen=True)
class FeatureMoments:
    count: int
    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def from_features(cls, features: np.ndarray) -> "FeatureMoments":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("features must have shape [N, D] with N >= 2")
        return cls(
            count=int(values.shape[0]),
            mean=values.mean(axis=0),
            covariance=np.cov(values, rowvar=False, ddof=1),
        )

    def merge(self, other: "FeatureMoments") -> "FeatureMoments":
        if self.mean.shape != other.mean.shape:
            raise ValueError("feature dimensions do not match")
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * (other.count / count)
        sum_squares = (
            (self.count - 1) * self.covariance
            + (other.count - 1) * other.covariance
            + np.outer(delta, delta) * (self.count * other.count / count)
        )
        return FeatureMoments(count, mean, sum_squares / (count - 1))


def calculate_fid(generated: FeatureMoments, reference: FeatureMoments) -> float:
    if generated.mean.shape != reference.mean.shape:
        raise ValueError("generated and reference feature dimensions do not match")
    mean_distance = np.square(generated.mean - reference.mean).sum()
    covariance_product = generated.covariance @ reference.covariance
    if np.count_nonzero(covariance_product) == 0:
        covariance_root = np.zeros_like(covariance_product)
    else:
        covariance_root, _ = scipy.linalg.sqrtm(covariance_product, disp=False)
    value = mean_distance + np.trace(
        generated.covariance + reference.covariance - 2.0 * covariance_root
    )
    return float(np.real(value))


def validate_image_set(images_root: Path, num_images: int) -> list[Path]:
    paths = sorted(images_root.rglob("*.png"))
    expected_names = {f"{index:06d}.png" for index in range(num_images)}
    actual_names = {path.name for path in paths}
    if len(paths) != num_images or actual_names != expected_names:
        raise ValueError(
            f"expected exactly {num_images} indexed images, found {len(paths)} in {images_root}"
        )
    return paths


def load_cached_fid(
    path: Path,
    *,
    generation_manifest_sha256: str,
    reference_sha256: str,
    detector_sha256: str,
) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "generation_manifest_sha256": generation_manifest_sha256,
        "reference_sha256": reference_sha256,
        "detector_sha256": detector_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"stale FID result does not match current image or metric assets: {path}")
    return float(payload["fid"])


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1)


def _moments_from_sums(count: int, total: np.ndarray, outer: np.ndarray) -> FeatureMoments:
    if count < 2:
        raise ValueError("at least two feature vectors are required")
    mean = total / count
    covariance = (outer - np.outer(mean, mean) * count) / (count - 1)
    return FeatureMoments(count, mean, covariance)


@torch.no_grad()
def calculate_feature_moments(
    image_paths: list[Path],
    detector: torch.nn.Module,
    *,
    total_batch: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> FeatureMoments:
    if total_batch % world_size != 0:
        raise ValueError("total_batch must be divisible by world_size")
    rank_paths = image_paths[rank::world_size]
    loader = torch.utils.data.DataLoader(
        _ImageDataset(rank_paths),
        batch_size=total_batch // world_size,
        shuffle=False,
        num_workers=3,
        prefetch_factor=2,
    )
    feature_sum: torch.Tensor | None = None
    feature_outer: torch.Tensor | None = None
    count = torch.zeros((), dtype=torch.int64, device=device)
    for images in loader:
        features = detector(images.to(device), return_features=True).to(torch.float64)
        if feature_sum is None:
            feature_sum = torch.zeros(features.shape[1], dtype=torch.float64, device=device)
            feature_outer = torch.zeros(
                (features.shape[1], features.shape[1]), dtype=torch.float64, device=device
            )
        feature_sum += features.sum(dim=0)
        feature_outer += features.T @ features
        count += features.shape[0]
    if feature_sum is None or feature_outer is None:
        raise ValueError("rank received no images for FID calculation")
    if world_size > 1:
        torch.distributed.all_reduce(count)
        torch.distributed.all_reduce(feature_sum)
        torch.distributed.all_reduce(feature_outer)
    return _moments_from_sums(
        int(count.item()),
        feature_sum.cpu().numpy(),
        feature_outer.cpu().numpy(),
    )


def _read_generation_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"method", "num_images", "model_sha256", "complete"}
    if not required.issubset(payload) or payload["complete"] is not True:
        raise ValueError(f"generation manifest is incomplete: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def run_fid(
    repo_root: Path,
    method: str,
    *,
    num_images: int,
    total_batch: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> float:
    root = repo_root.resolve()
    config = GenerationConfig(root, method, num_images, total_batch)
    assets = load_asset_manifest(root)
    generation_path = config.method_root / "generation_manifest.json"
    generation = _read_generation_manifest(generation_path)
    if generation["method"] != method or int(generation["num_images"]) != num_images:
        raise ValueError("generation manifest does not match requested FID run")
    if generation["model_sha256"] != assets.model.sha256:
        raise ValueError("generation manifest model does not match repository asset")
    generation_digest = sha256_file(generation_path)
    result_path = config.method_root / "fid.json"
    if result_path.exists():
        value = load_cached_fid(
            result_path,
            generation_manifest_sha256=generation_digest,
            reference_sha256=assets.reference.sha256,
            detector_sha256=assets.detector.sha256,
        )
        if world_size > 1:
            torch.distributed.barrier()
        return value

    image_paths = validate_image_set(config.images_root, num_images)
    with (root / assets.detector.path).open("rb") as handle:
        detector = pickle.load(handle).to(device).eval().requires_grad_(False)
    generated = calculate_feature_moments(
        image_paths,
        detector,
        total_batch=total_batch,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    with np.load(root / assets.reference.path) as reference_data:
        reference = FeatureMoments(
            num_images,
            np.asarray(reference_data["mu"], dtype=np.float64),
            np.asarray(reference_data["sigma"], dtype=np.float64),
        )
    value = calculate_fid(generated, reference)
    if rank == 0:
        _write_json_atomic(
            result_path,
            {
                "fid": value,
                "method": method,
                "num_images": num_images,
                "generation_manifest_sha256": generation_digest,
                "reference_sha256": assets.reference.sha256,
                "detector_sha256": assets.detector.sha256,
            },
        )
        print(f"FID ({method}): {value:.12g}", flush=True)
    if world_size > 1:
        torch.distributed.barrier()
    return value


def _init_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ImageNet FID")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return rank, world_size, torch.device("cuda", local_rank)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("baseline", "peek"), required=True)
    parser.add_argument("--num-images", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=1_024)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    rank, world_size, device = _init_distributed()
    try:
        run_fid(
            args.repo_root,
            args.method,
            num_images=args.num_images,
            total_batch=args.batch,
            device=device,
            rank=rank,
            world_size=world_size,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
