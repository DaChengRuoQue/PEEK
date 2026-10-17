from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.imagenet64_edm_fid5k import build_backend_command, summarize_result


class Imagenet64EdmFid5kTests(unittest.TestCase):
    def test_backend_command_is_one_fixed_compute_matched_pair(self) -> None:
        solver_root = Path("/tmp/diff-solvers-main")
        outdir = Path("/tmp/peek-result")

        command = build_backend_command(
            solver_root=solver_root,
            outdir=outdir,
            python_executable="python-test",
            num_images=5000,
            nproc=8,
            batch=1024,
            dry_run=False,
        )

        self.assertEqual(command[0:2], ["python-test", str(solver_root / "run_cifar_peek.py")])
        self.assertEqual(command[command.index("--dataset-name") + 1], "imagenet64")
        self.assertEqual(command[command.index("--nfe") + 1], "6")
        self.assertEqual(command[command.index("--rows") + 1], "dpmpp3m_peek")
        self.assertEqual(
            command[command.index("--fixed-peek-alpha-schedule") + 1],
            "1,1,1,0,0",
        )
        self.assertEqual(command[command.index("--final-max-order") + 1], "1")
        self.assertEqual(command[command.index("--num-final") + 1], "5000")
        self.assertEqual(command[command.index("--nproc") + 1], "8")
        self.assertEqual(command[command.index("--batch") + 1], "1024")
        self.assertEqual(command[command.index("--fid-batch") + 1], "1024")
        self.assertEqual(command[command.index("--outdir") + 1], str(outdir))
        self.assertNotIn("--skip-nfe-matched-baseline", command)
        self.assertNotIn("--peek-alphas", command)
        self.assertNotIn("--proxy-mode", command)

    def test_dry_run_is_forwarded_to_backend(self) -> None:
        command = build_backend_command(
            solver_root=Path("/tmp/diff-solvers-main"),
            outdir=Path("/tmp/peek-result"),
            python_executable="python-test",
            num_images=5000,
            nproc=8,
            batch=1024,
            dry_run=True,
        )

        self.assertEqual(command[-1], "--dry-run")

    def test_summary_reports_single_pair_reduction(self) -> None:
        backend_summary = {
            "dataset": "imagenet64",
            "base_nfe": 6,
            "num_final": 5000,
            "fixed_peek_alpha_schedule": "1,1,1,0,0",
            "results": [
                {
                    "row": "dpmpp3m_peek",
                    "method": "peek",
                    "actual_peek_nfe": 9,
                    "final_max_order": 1,
                    "final_fid": 10.0,
                    "nfe_matched_baseline_fid": 20.0,
                    "skip_nfe_matched_baseline": False,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "summary.json"
            destination = tmp_path / "reproduction_summary.json"
            source.write_text(json.dumps(backend_summary), encoding="utf-8")

            report = summarize_result(source, destination)

            self.assertEqual(report["baseline_fid"], 20.0)
            self.assertEqual(report["peek_fid"], 10.0)
            self.assertEqual(report["absolute_fid_reduction"], 10.0)
            self.assertEqual(report["relative_fid_reduction_percent"], 50.0)
            self.assertEqual(report["paper_relative_fid_reduction_percent"], 49.3)
            self.assertEqual(report["classification"], "roughly_reproduced")
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), report)

    def test_summary_rejects_nonmatching_protocol(self) -> None:
        invalid = {
            "dataset": "imagenet64",
            "base_nfe": 6,
            "num_final": 5000,
            "fixed_peek_alpha_schedule": "1,1,1,0,0",
            "results": [
                {
                    "row": "dpmpp3m_peek",
                    "method": "peek",
                    "actual_peek_nfe": 8,
                    "final_max_order": 1,
                    "final_fid": 10.0,
                    "nfe_matched_baseline_fid": 20.0,
                    "skip_nfe_matched_baseline": False,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "summary.json"
            source.write_text(json.dumps(invalid), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "actual PEEK NFE"):
                summarize_result(source, tmp_path / "out.json")


if __name__ == "__main__":
    unittest.main()
