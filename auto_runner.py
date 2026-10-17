#!/usr/bin/env python3
"""Execute command files sequentially and retain one timestamped log per command."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def read_commands(path: Path) -> list[str]:
    commands = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            commands.append(line)
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=Path, nargs="+", required=True)
    parser.add_argument("--log_dir", type=Path, default=Path("runner_logs"))
    parser.add_argument("--stop_on_error", action="store_true")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    queued = [(source, command) for source in args.commands for command in read_commands(source)]
    if not queued:
        raise SystemExit("no commands found")
    for index, (source, command) in enumerate(queued):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = args.log_dir / f"{index:03d}_{timestamp}.log"
        print(f"[{index + 1}/{len(queued)}] {command}", flush=True)
        print(f"log: {log_path.resolve()}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# source: {source}\n{command}\n\n")
            log.flush()
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            print(f"command failed with exit code {result.returncode}: {command}", flush=True)
            if args.stop_on_error:
                return result.returncode
    print("all commands finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
