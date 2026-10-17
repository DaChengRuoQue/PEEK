from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from peek.imagenet64.assets import (
    AssetSpec,
    install_asset,
    load_asset_manifest,
    prepare_assets,
    sha256_file,
)


class Imagenet64AssetTests(unittest.TestCase):
    def test_preparation_script_runs_directly_from_repository_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "scripts/prepare_imagenet64_assets.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint-source", result.stdout)

    def test_sha256_file_matches_known_literal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.bin"
            path.write_bytes(b"abc")

            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_install_asset_is_atomic_and_rejects_wrong_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            destination = root / "assets" / "target.bin"
            source.write_bytes(b"wrong")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                install_asset(str(source), destination, "0" * 64)

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".bin.part").exists())

    def test_prepare_and_load_assets_stay_inside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            sources = Path(tmp) / "sources"
            repo.mkdir()
            sources.mkdir()
            payloads = {
                "model": b"model-fixture",
                "detector": b"detector-fixture",
                "reference": b"reference-fixture",
            }
            specs = []
            for name, payload in payloads.items():
                source = sources / f"{name}.bin"
                source.write_bytes(payload)
                specs.append(
                    AssetSpec(
                        name=name,
                        filename=f"{name}.bin",
                        source=str(source),
                        sha256=hashlib.sha256(payload).hexdigest(),
                    )
                )

            manifest = prepare_assets(repo, asset_specs=tuple(specs))
            loaded = load_asset_manifest(repo)

            self.assertEqual(loaded, manifest)
            for asset in (loaded.model, loaded.detector, loaded.reference):
                resolved = (repo / asset.path).resolve()
                self.assertTrue(resolved.is_relative_to(repo.resolve()))
                self.assertEqual(sha256_file(resolved), asset.sha256)

    def test_load_rejects_incomplete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            asset_dir = repo / "checkpoints" / "imagenet64"
            asset_dir.mkdir(parents=True)
            (asset_dir / "manifest.json").write_text(
                json.dumps({"version": 1, "assets": {"model": {}}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "complete model, detector, and reference"):
                load_asset_manifest(repo)

    def test_load_rejects_manifest_path_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            outside = Path(tmp) / "outside.bin"
            outside.write_bytes(b"outside")
            digest = sha256_file(outside)
            asset_dir = repo / "checkpoints" / "imagenet64"
            asset_dir.mkdir(parents=True)
            entry = {"path": "../outside.bin", "sha256": digest, "size": outside.stat().st_size}
            (asset_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "assets": {"model": entry, "detector": entry, "reference": entry},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outside repository"):
                load_asset_manifest(repo)


if __name__ == "__main__":
    unittest.main()
