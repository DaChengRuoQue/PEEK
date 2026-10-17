from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from peek.imagenet64.generate import (
    GenerationConfig,
    generate_rank,
    generation_settings,
    prepare_generation_directory,
    save_png_atomic,
    shard_indices,
)
from peek.imagenet64.random import StackedRandomGenerator


class TinyDenoiser:
    img_channels = 3
    img_resolution = 2
    label_dim = 4

    def __call__(
        self,
        state: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if class_labels is None or class_labels.shape[1] != self.label_dim:
            raise AssertionError("invalid class labels")
        return 0.01 * state


class Imagenet64GenerateTests(unittest.TestCase):
    def test_indexed_random_stream_is_reproducible_for_latents_and_labels(self) -> None:
        seeds = [7, 19]
        first = StackedRandomGenerator(torch.device("cpu"), seeds)
        second = StackedRandomGenerator(torch.device("cpu"), seeds)

        first_latents = first.randn((2, 3, 2, 2), dtype=torch.float32, device=torch.device("cpu"))
        first_labels = first.one_hot_labels(4, dtype=torch.float32, device=torch.device("cpu"))
        second_latents = second.randn((2, 3, 2, 2), dtype=torch.float32, device=torch.device("cpu"))
        second_labels = second.one_hot_labels(4, dtype=torch.float32, device=torch.device("cpu"))

        torch.testing.assert_close(first_latents, second_latents, rtol=0, atol=0)
        torch.testing.assert_close(first_labels, second_labels, rtol=0, atol=0)
        self.assertTrue(torch.equal(first_labels.sum(dim=1), torch.ones(2)))

    def test_sharding_covers_uneven_count_once(self) -> None:
        shards = [shard_indices(11, rank, 4) for rank in range(4)]

        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(11)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(shards[0], [0, 4, 8])
        self.assertEqual(shards[3], [3, 7])

    def test_generation_settings_select_only_fixed_baseline_or_peek_path(self) -> None:
        self.assertEqual(generation_settings("baseline"), (9, ()))
        self.assertEqual(generation_settings("peek"), (6, (1, 1, 1, 0, 0)))
        with self.assertRaisesRegex(ValueError, "baseline or peek"):
            generation_settings("search")

    def test_png_write_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "nested" / "000007.png"
            pixels = np.array(
                [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
                dtype=np.uint8,
            )

            save_png_atomic(destination, pixels)

            self.assertTrue(destination.is_file())
            self.assertFalse(destination.with_suffix(".png.part").exists())
            np.testing.assert_array_equal(np.array(Image.open(destination)), pixels)

    def test_partial_output_with_mismatched_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = GenerationConfig(Path(tmp), "baseline", num_images=3, total_batch=2)
            manifest = prepare_generation_directory(config, model_sha256="a" * 64)
            path = config.method_root / "generation_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["model_sha256"] = "b" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_generation_directory(config, model_sha256="a" * 64)
            self.assertEqual(manifest.model_sha256, "a" * 64)

    def test_generate_rank_writes_all_images_and_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = GenerationConfig(Path(tmp), "peek", num_images=3, total_batch=2)

            manifest = generate_rank(
                config,
                net=TinyDenoiser(),
                model_sha256="c" * 64,
                rank=0,
                world_size=1,
                device=torch.device("cpu"),
            )

            self.assertTrue(manifest.complete)
            self.assertEqual(manifest.num_images, 3)
            images = sorted(config.images_root.rglob("*.png"))
            self.assertEqual([path.name for path in images], ["000000.png", "000001.png", "000002.png"])
            stored = json.loads((config.method_root / "generation_manifest.json").read_text())
            self.assertTrue(stored["complete"])


if __name__ == "__main__":
    unittest.main()
