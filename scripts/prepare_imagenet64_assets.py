#!/usr/bin/env python3
"""Prepare pinned NVIDIA assets inside the PEEK repository."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peek.imagenet64.assets import prepare_assets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-source", help="Optional local copy of the pinned model checkpoint")
    args = parser.parse_args()
    manifest = prepare_assets(REPO_ROOT, checkpoint_source=args.checkpoint_source)
    print(json.dumps(asdict(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
