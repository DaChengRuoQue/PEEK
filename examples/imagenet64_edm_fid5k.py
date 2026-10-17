"""Launch one compute-matched ImageNet-64 EDM PEEK FID comparison."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PAPER_REDUCTION_PERCENT = 49.3
ROUGH_TOLERANCE_PERCENTAGE_POINTS = 10.0
FIXED_SCHEDULE = "1,1,1,0,0"


def build_backend_command(
    *,
    solver_root: Path,
    outdir: Path,
    python_executable: str,
    num_images: int,
    nproc: int,
    batch: int,
    dry_run: bool,
) -> list[str]:
    if num_images != 5000:
        raise ValueError("this reproduction is fixed to one 5,000-image evaluation per method")
    if nproc < 1:
        raise ValueError("nproc must be positive")
    if batch < 1:
        raise ValueError("batch must be positive")

    command = [
        python_executable,
        str(solver_root / "run_cifar_peek.py"),
        "--dataset-name",
        "imagenet64",
        "--nfe",
        "6",
        "--rows",
        "dpmpp3m_peek",
        "--fixed-peek-alpha-schedule",
        FIXED_SCHEDULE,
        "--final-max-order",
        "1",
        "--num-final",
        str(num_images),
        "--nproc",
        str(nproc),
        "--batch",
        str(batch),
        "--fid-batch",
        str(batch),
        "--preview",
        "0",
        "--outdir",
        str(outdir),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _require_protocol(summary: dict) -> dict:
    if summary.get("dataset") != "imagenet64":
        raise ValueError("backend summary is not for ImageNet-64")
    if summary.get("base_nfe") != 6:
        raise ValueError("backend summary does not use base NFE 6")
    if summary.get("num_final") != 5000:
        raise ValueError("backend summary is not the requested 5k evaluation")
    if summary.get("fixed_peek_alpha_schedule") != FIXED_SCHEDULE:
        raise ValueError("backend summary does not use the fixed paper schedule")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("backend summary must contain exactly one paired result")
    result = results[0]
    if result.get("row") != "dpmpp3m_peek" or result.get("method") != "peek":
        raise ValueError("backend summary is not the first-order DPM-Solver++ PEEK row")
    if result.get("actual_peek_nfe") != 9:
        raise ValueError("backend summary has an unexpected actual PEEK NFE")
    if result.get("final_max_order") != 1:
        raise ValueError("backend summary is not first-order DPM-Solver++")
    if result.get("skip_nfe_matched_baseline") is not False:
        raise ValueError("backend summary skipped the compute-matched baseline")
    return result


def summarize_result(source: Path, destination: Path) -> dict:
    summary = json.loads(source.read_text(encoding="utf-8"))
    result = _require_protocol(summary)
    peek_fid = float(result["final_fid"])
    baseline_fid = float(result["nfe_matched_baseline_fid"])
    if baseline_fid <= 0.0:
        raise ValueError("baseline FID must be positive")

    absolute_reduction = baseline_fid - peek_fid
    relative_reduction = 100.0 * absolute_reduction / baseline_fid
    roughly_reproduced = (
        absolute_reduction > 0.0
        and abs(relative_reduction - PAPER_REDUCTION_PERCENT)
        <= ROUGH_TOLERANCE_PERCENTAGE_POINTS
    )
    report = {
        "evaluation": "one_paired_fid5k",
        "dataset": "imagenet64",
        "model": "edm-imagenet-64x64-cond-adm",
        "solver": "first-order DPM-Solver++",
        "base_nfe": 6,
        "matched_nfe": 9,
        "peek_schedule": [1, 1, 1, 0, 0],
        "baseline_fid": baseline_fid,
        "peek_fid": peek_fid,
        "absolute_fid_reduction": absolute_reduction,
        "relative_fid_reduction_percent": relative_reduction,
        "paper_relative_fid_reduction_percent": PAPER_REDUCTION_PERCENT,
        "classification": "roughly_reproduced" if roughly_reproduced else "not_roughly_reproduced",
        "backend_summary": str(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    solver_root = args.solver_root.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    runner = solver_root / "run_cifar_peek.py"
    checkpoint = solver_root / "src/imagenet64/edm-imagenet-64x64-cond-adm.pkl"
    if not runner.is_file():
        raise SystemExit(f"missing solver backend: {runner}")
    if not args.dry_run and not checkpoint.is_file():
        raise SystemExit(f"missing pretrained checkpoint: {checkpoint}")

    command = build_backend_command(
        solver_root=solver_root,
        outdir=outdir,
        python_executable=sys.executable,
        num_images=args.num_images,
        nproc=args.nproc,
        batch=args.batch,
        dry_run=args.dry_run,
    )
    print("Backend command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=solver_root, check=True)
    if args.dry_run:
        return 0

    report = summarize_result(outdir / "summary.json", outdir / "reproduction_summary.json")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
