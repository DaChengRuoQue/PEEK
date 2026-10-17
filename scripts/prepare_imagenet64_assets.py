#!/usr/bin/env python3
"""Prepare pinned NVIDIA assets inside the PEEK repository."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from peek.imagenet64.assets import prepare_assets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-source", help="Optional local copy of the pinned model checkpoint")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = prepare_assets(repo_root, checkpoint_source=args.checkpoint_source)
    print(json.dumps(asdict(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
