"""Deterministic distributed generation for the ImageNet-64 comparison."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from .assets import load_asset_manifest
from .protocol import paper_protocol
from .random import StackedRandomGenerator
from .sampler import sample_dpmpp_first_order


Method = Literal["baseline", "peek"]


def generation_settings(method: str) -> tuple[int, tuple[int, ...]]:
    protocol = paper_protocol()
    if method == "baseline":
        return protocol.baseline_nfe, ()
    if method == "peek":
        return protocol.peek_base_nfe, protocol.peek_gates
    raise ValueError("method must be baseline or peek")


def shard_indices(num_images: int, rank: int, world_size: int) -> list[int]:
    if num_images < 1:
        raise ValueError("num_images must be positive")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed rank or world size")
    return list(range(rank, num_images, world_size))


@dataclass(frozen=True)
class GenerationConfig:
    repo_root: Path
    method: Method
    num_images: int = 50_000
    total_batch: int = 768

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", self.repo_root.resolve())
        generation_settings(self.method)
        if self.num_images < 1:
            raise ValueError("num_images must be positive")
        if self.total_batch < 1:
            raise ValueError("total_batch must be positive")

    @property
    def method_root(self) -> Path:
        return self.repo_root / "runs" / "imagenet64_edm_fid50k" / self.method

    @property
    def images_root(self) -> Path:
        return self.method_root / "images"


@dataclass(frozen=True)
class GenerationManifest:
    version: int
    method: str
    num_images: int
    model_sha256: str
    base_nfe: int
    gates: tuple[int, ...]
    sigma_min: float
    sigma_max: float
    indexed_stream: str
    complete: bool


def _manifest_path(config: GenerationConfig) -> Path:
    return config.method_root / "generation_manifest.json"


def _expected_manifest(config: GenerationConfig, model_sha256: str) -> GenerationManifest:
    base_nfe, gates = generation_settings(config.method)
    protocol = paper_protocol()
    return GenerationManifest(
        version=1,
        method=config.method,
        num_images=config.num_images,
        model_sha256=model_sha256,
        base_nfe=base_nfe,
        gates=gates,
        sigma_min=protocol.sigma_min,
        sigma_max=protocol.sigma_max,
        indexed_stream="seed-indexed-latent-and-label-v1",
        complete=False,
    )


def _write_manifest(path: Path, manifest: GenerationManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _read_manifest(path: Path) -> GenerationManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["gates"] = tuple(int(value) for value in payload["gates"])
        return GenerationManifest(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid generation manifest: {path}") from exc


def prepare_generation_directory(
    config: GenerationConfig, model_sha256: str
) -> GenerationManifest:
    expected = _expected_manifest(config, model_sha256)
    path = _manifest_path(config)
    if path.exists():
        existing = _read_manifest(path)
        if replace(existing, complete=False) != expected:
            raise ValueError(f"existing generation manifest does not match requested run: {path}")
        return existing
    if config.images_root.exists() and any(config.images_root.rglob("*.png")):
        raise ValueError("partial output exists without a generation manifest")
    config.images_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(path, expected)
    return expected


def save_png_atomic(destination: Path, pixels: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    Image.fromarray(pixels, "RGB").save(partial, format="PNG")
    os.replace(partial, destination)


def _image_path(root: Path, index: int) -> Path:
    return root / f"{index - index % 1000:06d}" / f"{index:06d}.png"


def _load_network(repo_root: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    assets = load_asset_manifest(repo_root)
    model_path = repo_root / assets.model.path
    with model_path.open("rb") as handle:
        network = pickle.load(handle)["ema"].to(device)
    network.eval().requires_grad_(False)
    return network, assets.model.sha256


def _distributed_barrier(world_size: int) -> None:
    if world_size > 1:
        torch.distributed.barrier()


def generate_rank(
    config: GenerationConfig,
    *,
    net: torch.nn.Module | object | None = None,
    model_sha256: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device | None = None,
) -> GenerationManifest:
    device = device or torch.device("cuda")
    if config.total_batch % world_size != 0:
        raise ValueError("total_batch must be divisible by world_size")
    per_rank_batch = config.total_batch // world_size
    if net is None:
        net, loaded_digest = _load_network(config.repo_root, device)
        model_sha256 = loaded_digest
    if model_sha256 is None:
        raise ValueError("model_sha256 is required when supplying a network")

    if rank == 0:
        prepare_generation_directory(config, model_sha256)
    _distributed_barrier(world_size)
    if rank != 0:
        prepare_generation_directory(config, model_sha256)

    base_nfe, gates = generation_settings(config.method)
    indices = shard_indices(config.num_images, rank, world_size)
    for offset in range(0, len(indices), per_rank_batch):
        batch_indices = indices[offset : offset + per_rank_batch]
        missing = [index for index in batch_indices if not _image_path(config.images_root, index).is_file()]
        if not missing:
            continue
        random = StackedRandomGenerator(device, missing)
        latents = random.randn(
            (len(missing), int(net.img_channels), int(net.img_resolution), int(net.img_resolution)),
            dtype=torch.float32,
            device=device,
        )
        labels = random.one_hot_labels(int(net.label_dim), dtype=torch.float32, device=device)
        images = sample_dpmpp_first_order(
            net,
            latents,
            labels,
            num_evaluations=base_nfe,
            gates=gates,
        )
        pixels = (
            (images * 127.5 + 128.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        for index, image in zip(missing, pixels):
            save_png_atomic(_image_path(config.images_root, index), image)

    _distributed_barrier(world_size)
    if rank == 0:
        image_paths = list(config.images_root.rglob("*.png"))
        expected_names = {f"{index:06d}.png" for index in range(config.num_images)}
        actual_names = {path.name for path in image_paths}
        if len(image_paths) != config.num_images or actual_names != expected_names:
            raise RuntimeError(
                f"generated image set is incomplete: found {len(image_paths)}, expected {config.num_images}"
            )
        completed = replace(_expected_manifest(config, model_sha256), complete=True)
        _write_manifest(_manifest_path(config), completed)
    _distributed_barrier(world_size)
    return _read_manifest(_manifest_path(config))


def _init_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ImageNet generation")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return rank, world_size, torch.device("cuda", local_rank)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("baseline", "peek"), required=True)
    parser.add_argument("--num-images", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=768)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    rank, world_size, device = _init_distributed()
    try:
        config = GenerationConfig(args.repo_root, args.method, args.num_images, args.batch)
        manifest = generate_rank(config, rank=rank, world_size=world_size, device=device)
        if rank == 0:
            print(json.dumps(asdict(manifest), indent=2), flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
