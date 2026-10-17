from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from peek.imagenet64.evaluate import (
    build_evaluation_commands,
    summarize_result,
    validate_repo_command,
)
from peek.imagenet64.protocol import paper_protocol


def launched_module(command: list[str]) -> str:
    positions = [index for index, value in enumerate(command) if value == "-m"]
    return command[positions[-1] + 1]


class Imagenet64EdmFid50kTests(unittest.TestCase):
    def test_builds_exactly_one_ordered_fixed_pair(self) -> None:
        repo = Path("/tmp/peek-repo").resolve()

        commands = build_evaluation_commands(
            repo,
            paper_protocol(),
            nproc=6,
            batch=768,
            python_executable=sys.executable,
        )

        self.assertEqual(len(commands), 4)
        self.assertEqual(
            [launched_module(command) for command in commands],
            [
                "peek.imagenet64.generate",
                "peek.imagenet64.fid",
                "peek.imagenet64.generate",
                "peek.imagenet64.fid",
            ],
        )
        self.assertEqual(
            [command[command.index("--method") + 1] for command in commands],
            ["baseline", "baseline", "peek", "peek"],
        )
        for command in commands:
            self.assertIn("--nproc_per_node=6", command)
            self.assertEqual(command[command.index("--num-images") + 1], "50000")
            self.assertEqual(command[command.index("--batch") + 1], "768")
            self.assertEqual(command[command.index("--repo-root") + 1], str(repo))
            removed_external_option = "--" + "solver" + "-root"
            self.assertNotIn(removed_external_option, command)
            self.assertFalse(any("search" in value.lower() for value in command))

    def test_command_boundary_rejects_external_runtime_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            command = [
                sys.executable,
                "-m",
                "peek.imagenet64.generate",
                "--repo-root",
                "/tmp/not-the-repository",
            ]

            with self.assertRaisesRegex(ValueError, "outside repository"):
                validate_repo_command(repo, command, python_executable=sys.executable)

    def test_summary_reports_relative_reduction_from_the_single_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = repo / "runs" / "imagenet64_edm_fid50k"
            for method, value in (("baseline", 20.0), ("peek", 10.0)):
                method_root = run_root / method
                method_root.mkdir(parents=True)
                (method_root / "fid.json").write_text(
                    json.dumps({"fid": value, "method": method, "num_images": 50_000}),
                    encoding="utf-8",
                )

            report = summarize_result(repo)

            self.assertEqual(report["baseline_fid"], 20.0)
            self.assertEqual(report["peek_fid"], 10.0)
            self.assertEqual(report["absolute_fid_reduction"], 10.0)
            self.assertEqual(report["relative_fid_reduction_percent"], 50.0)
            self.assertEqual(report["classification"], "roughly_reproduced")
            stored = json.loads((run_root / "reproduction_summary.json").read_text())
            self.assertEqual(stored, report)

    def test_summary_rejects_wrong_method_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = repo / "runs" / "imagenet64_edm_fid50k"
            for directory, method in (("baseline", "peek"), ("peek", "peek")):
                method_root = run_root / directory
                method_root.mkdir(parents=True)
                (method_root / "fid.json").write_text(
                    json.dumps({"fid": 10.0, "method": method, "num_images": 50_000}),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "baseline FID metadata"):
                summarize_result(repo)


if __name__ == "__main__":
    unittest.main()
