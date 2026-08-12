#!/usr/bin/env python3
"""Require every delayed Slurm job to execute the exact clean submitted revision."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if len(args.expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.expected_commit
    ):
        raise ValueError("AIME_EXPECTED_GIT_COMMIT is not a full lowercase Git SHA")
    root = args.repo_root.resolve()
    head = output("git", "-C", str(root), "rev-parse", "HEAD")
    if head != args.expected_commit:
        raise RuntimeError(
            f"runtime Git revision changed after submission: expected={args.expected_commit} actual={head}"
        )
    status = output(
        "git",
        "-C",
        str(root),
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(f"runtime Git worktree is not clean:\n{status}")
    print(f"AIME_RUNTIME_REPO_GATE_OK commit={head}", flush=True)


if __name__ == "__main__":
    main()
