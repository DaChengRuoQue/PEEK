from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from peek.imagenet64.fid import (
    FeatureMoments,
    calculate_fid,
    load_cached_fid,
    validate_image_set,
)


class Imagenet64FidTests(unittest.TestCase):
    def test_feature_moment_merge_matches_direct_accumulation(self) -> None:
        features = np.array(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 8.0], [9.0, 2.0]], dtype=np.float64
        )

        direct = FeatureMoments.from_features(features)
        merged = FeatureMoments.from_features(features[:2]).merge(
            FeatureMoments.from_features(features[2:])
        )

        self.assertEqual(merged.count, 4)
        np.testing.assert_allclose(merged.mean, direct.mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(merged.covariance, direct.covariance, rtol=0, atol=1e-12)

    def test_fid_matches_hand_checked_two_dimensional_fixture(self) -> None:
        generated = FeatureMoments.from_features(
            np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        )
        reference = FeatureMoments.from_features(
            np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        )

        self.assertAlmostEqual(calculate_fid(generated, reference), 6.0, places=12)

    def test_validate_image_set_requires_exact_indexed_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "000000.png").write_bytes(b"fixture")
            (root / "000002.png").write_bytes(b"fixture")

            with self.assertRaisesRegex(ValueError, "exactly 3 indexed images"):
                validate_image_set(root, 3)

    def test_cached_fid_rejects_stale_generation_or_reference_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fid.json"
            path.write_text(
                json.dumps(
                    {
                        "fid": 12.5,
                        "generation_manifest_sha256": "a" * 64,
                        "reference_sha256": "b" * 64,
                        "detector_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale FID result"):
                load_cached_fid(
                    path,
                    generation_manifest_sha256="d" * 64,
                    reference_sha256="b" * 64,
                    detector_sha256="c" * 64,
                )

    def test_cached_fid_returns_matching_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fid.json"
            path.write_text(
                json.dumps(
                    {
                        "fid": 12.5,
                        "generation_manifest_sha256": "a" * 64,
                        "reference_sha256": "b" * 64,
                        "detector_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_cached_fid(
                    path,
                    generation_manifest_sha256="a" * 64,
                    reference_sha256="b" * 64,
                    detector_sha256="c" * 64,
                ),
                12.5,
            )


if __name__ == "__main__":
    unittest.main()
