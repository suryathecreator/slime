#!/usr/bin/env python3
"""Require every delayed Slurm job to execute the exact clean submitted revision."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def validate_revision(root: Path, expected_commit: str, allowed_descendant_path: str | None = None) -> str:
    head = output("git", "-C", str(root), "rev-parse", "HEAD")
    if head == expected_commit:
        return head
    if allowed_descendant_path is None:
        raise RuntimeError(f"runtime Git revision changed after submission: expected={expected_commit} actual={head}")
    allowed = Path(allowed_descendant_path)
    if allowed.is_absolute() or ".." in allowed.parts or allowed.as_posix() != allowed_descendant_path:
        raise ValueError(f"invalid allowed descendant path: {allowed_descendant_path}")
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", expected_commit, head],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    changed = set(output("git", "-C", str(root), "diff", "--name-only", expected_commit, head, "--").splitlines())
    if ancestor.returncode != 0 or changed != {allowed_descendant_path}:
        raise RuntimeError(
            "runtime Git revision changed outside the allowed post-submission metadata path: "
            f"expected={expected_commit} actual={head} changed={sorted(changed)}"
        )
    return head


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
    allowed_descendant_path = os.environ.get("AIME_RUNTIME_ALLOWED_DESCENDANT_PATH") or None
    head = validate_revision(root, args.expected_commit, allowed_descendant_path)
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
    print(
        f"AIME_RUNTIME_REPO_GATE_OK expected={args.expected_commit} actual={head} "
        f"allowed_descendant_path={allowed_descendant_path or 'none'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
