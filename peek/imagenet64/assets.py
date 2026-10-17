"""Preparation and validation of repository-local ImageNet evaluation assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MODEL_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl"
DETECTOR_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
REFERENCE_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz"

DEFAULT_ASSET_SPECS = (
    # Digests are for the public NVIDIA files at the URLs above.
    ("model", "edm-imagenet-64x64-cond-adm.pkl", MODEL_URL, "a92c856457388fca0ac9be6b3b675ecfd68b90dfd90e7860b05e20c65fdb864f"),
    ("detector", "inception-2015-12-05.pkl", DETECTOR_URL, "a31bd1d4522101109044ec28621a5fcb591ed6115e0d58104fc013fa01ef94f7"),
    ("reference", "imagenet-64x64.npz", REFERENCE_URL, "6e0e6f8ae9d9c9e3a96d43ff4f40b5866c2de63430266dfe8f53210c5738d5fd"),
)


@dataclass(frozen=True)
class AssetSpec:
    name: str
    filename: str
    source: str
    sha256: str


@dataclass(frozen=True)
class AssetRecord:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class AssetManifest:
    version: int
    model: AssetRecord
    detector: AssetRecord
    reference: AssetRecord


def default_asset_specs() -> tuple[AssetSpec, ...]:
    return tuple(AssetSpec(*values) for values in DEFAULT_ASSET_SPECS)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_inside(repo_root: Path, path: Path) -> Path:
    root = repo_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"asset path resolves outside repository: {path}")
    return resolved


def install_asset(source: str, destination: Path, expected_sha256: str) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for existing {destination}: expected {expected_sha256}, got {actual}"
            )
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        if source.startswith(("https://", "http://")):
            with urllib.request.urlopen(source) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        else:
            shutil.copyfile(Path(source).expanduser().resolve(), partial)
        actual = sha256_file(partial)
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {source}: expected {expected_sha256}, got {actual}"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: AssetManifest) -> None:
    payload = {
        "version": manifest.version,
        "assets": {
            "model": asdict(manifest.model),
            "detector": asdict(manifest.detector),
            "reference": asdict(manifest.reference),
        },
    }
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def prepare_assets(
    repo_root: Path,
    checkpoint_source: str | None = None,
    *,
    asset_specs: Iterable[AssetSpec] | None = None,
) -> AssetManifest:
    root = repo_root.resolve()
    asset_dir = _require_inside(root, root / "checkpoints" / "imagenet64")
    asset_dir.mkdir(parents=True, exist_ok=True)
    specs = tuple(asset_specs) if asset_specs is not None else default_asset_specs()
    if checkpoint_source is not None:
        specs = tuple(
            AssetSpec(spec.name, spec.filename, checkpoint_source, spec.sha256)
            if spec.name == "model"
            else spec
            for spec in specs
        )
    if {spec.name for spec in specs} != {"model", "detector", "reference"}:
        raise ValueError("asset specifications must contain complete model, detector, and reference entries")

    records: dict[str, AssetRecord] = {}
    for spec in specs:
        destination = _require_inside(root, asset_dir / spec.filename)
        install_asset(spec.source, destination, spec.sha256)
        records[spec.name] = AssetRecord(
            path=destination.relative_to(root).as_posix(),
            sha256=spec.sha256,
            size=destination.stat().st_size,
        )
    manifest = AssetManifest(version=1, **records)
    _write_manifest(asset_dir / "manifest.json", manifest)
    return manifest


def load_asset_manifest(repo_root: Path) -> AssetManifest:
    root = repo_root.resolve()
    manifest_path = _require_inside(root, root / "checkpoints" / "imagenet64" / "manifest.json")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assets = payload["assets"]
        manifest = AssetManifest(
            version=int(payload["version"]),
            model=AssetRecord(**assets["model"]),
            detector=AssetRecord(**assets["detector"]),
            reference=AssetRecord(**assets["reference"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("asset manifest must contain complete model, detector, and reference entries") from exc
    if manifest.version != 1:
        raise ValueError(f"unsupported asset manifest version: {manifest.version}")
    for record in (manifest.model, manifest.detector, manifest.reference):
        path = _require_inside(root, root / record.path)
        if not path.is_file():
            raise ValueError(f"asset is missing: {record.path}")
        actual = sha256_file(path)
        if actual != record.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {record.path}: expected {record.sha256}, got {actual}"
            )
        if path.stat().st_size != record.size:
            raise ValueError(f"size mismatch for {record.path}")
    return manifest
