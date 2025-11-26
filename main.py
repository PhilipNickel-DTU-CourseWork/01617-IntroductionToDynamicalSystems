#!/usr/bin/env python
"""
Sync Assignment 2 figures into the Overleaf project.

Copies `Assignment_2/figures` to `Assignment_2/TexReport/overleaf/figures`,
replacing the target.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "Assignment_2" / "figures"
    dst = repo_root / "Assignment_2" / "TexReport" / "overleaf" / "figures"

    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        return 1

    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    print(f"Copied figures from {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
