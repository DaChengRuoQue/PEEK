"""Orchestrate one fixed, compute-matched ImageNet-64 FID comparison."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .protocol import EvaluationProtocol, paper_protocol


PAPER_REDUCTION_PERCENT = 49.3
ROUGH_TOLERANCE_PERCENTAGE_POINTS = 10.0
ALLOWED_MODULES = {"peek.imagenet64.generate", "peek.imagenet64.fid"}


def validate_repo_command(
    repo_root: Path,
    command: list[str],
    *,
    python_executable: str = sys.executable,
) -> None:
    root = repo_root.resolve()
    if not command or Path(command[0]).resolve() != Path(python_executable).resolve():
        raise ValueError("evaluation subprocess must use the active Python executable")
    module_positions = [index for index, value in enumerate(command) if value == "-m"]
    if not module_positions or module_positions[-1] + 1 >= len(command):
        raise ValueError("evaluation subprocess must launch a repository module")
    module = command[module_positions[-1] + 1]
    if module not in ALLOWED_MODULES:
        raise ValueError(f"unsupported evaluation module: {module}")
    for token in command[1:]:
        candidate = token.split("=", 1)[-1] if "=" in token else token
        if candidate.startswith("/") and not Path(candidate).resolve().is_relative_to(root):
            raise ValueError(f"runtime path resolves outside repository: {candidate}")


def build_evaluation_commands(
    repo_root: Path,
    protocol: EvaluationProtocol,
    *,
    nproc: int,
    batch: int,
    python_executable: str = sys.executable,
) -> tuple[list[str], ...]:
    root = repo_root.resolve()
    if protocol != paper_protocol():
        raise ValueError("only the fixed paper-derived protocol is supported")
    if nproc < 1 or batch < 1 or batch % nproc != 0:
        raise ValueError("batch must be positive and divisible by nproc")

    commands: list[list[str]] = []
    for method in ("baseline", "peek"):
        common = [
            python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            "-m",
        ]
        commands.append(
            common
            + [
                "peek.imagenet64.generate",
                "--method",
                method,
                "--num-images",
                str(protocol.num_images),
                "--batch",
                str(batch),
                "--repo-root",
                str(root),
            ]
        )
        commands.append(
            common
            + [
                "peek.imagenet64.fid",
                "--method",
                method,
                "--num-images",
                str(protocol.num_images),
                "--batch",
                str(batch),
                "--repo-root",
                str(root),
            ]
        )
    result = tuple(commands)
    for command in result:
        validate_repo_command(root, command, python_executable=python_executable)
    return result


def _load_fid(path: Path, method: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") != method or payload.get("num_images") != 50_000:
        raise ValueError(f"{method} FID metadata does not match the fixed evaluation")
    value = float(payload["fid"])
    if value <= 0.0:
        raise ValueError(f"{method} FID must be positive")
    return value


def summarize_result(repo_root: Path) -> dict:
    run_root = repo_root.resolve() / "runs" / "imagenet64_edm_fid50k"
    baseline_fid = _load_fid(run_root / "baseline" / "fid.json", "baseline")
    peek_fid = _load_fid(run_root / "peek" / "fid.json", "peek")
    absolute_reduction = baseline_fid - peek_fid
    relative_reduction = 100.0 * absolute_reduction / baseline_fid
    roughly_reproduced = absolute_reduction > 0.0 and abs(
        relative_reduction - PAPER_REDUCTION_PERCENT
    ) <= ROUGH_TOLERANCE_PERCENTAGE_POINTS
    report = {
        "evaluation": "one_paired_fid50k",
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
    }
    destination = run_root / "reproduction_summary.json"
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, destination)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc", type=int, default=6)
    parser.add_argument("--batch", type=int, default=768)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    protocol = paper_protocol()
    commands = build_evaluation_commands(
        repo_root,
        protocol,
        nproc=args.nproc,
        batch=args.batch,
    )
    print(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "num_images": protocol.num_images,
                "baseline_nfe": protocol.baseline_nfe,
                "peek_base_nfe": protocol.peek_base_nfe,
                "peek_gates": protocol.peek_gates,
                "peek_total_nfe": protocol.peek_total_nfe,
            },
            indent=2,
        ),
        flush=True,
    )
    for command in commands:
        print("$ " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repo_root, check=True)
    if not args.dry_run:
        print(json.dumps(summarize_result(repo_root), indent=2), flush=True)
    return 0
