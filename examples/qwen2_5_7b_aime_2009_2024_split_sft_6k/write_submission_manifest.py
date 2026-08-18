#!/usr/bin/env python3
"""Write the immutable ten-job training-chain manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.data_utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-worktree", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry_run", "in_progress", "submitted"), required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        raise FileExistsError(f"refusing to overwrite submission manifest: {args.output}")
    with args.jobs_tsv.open(newline="", encoding="utf-8") as handle:
        jobs: list[dict[str, Any]] = list(csv.DictReader(handle, delimiter="\t"))
    expected = 10 if args.mode in ("dry_run", "submitted") else None
    if expected is not None and len(jobs) != expected:
        raise ValueError(f"{args.mode} manifest must contain exactly {expected} jobs")
    if not jobs:
        raise ValueError("submission manifest cannot be empty")
    for index, job in enumerate(jobs):
        if args.mode != "dry_run" and not str(job["job_id"]).isdigit():
            raise ValueError(f"nonnumeric submitted job ID: {job['job_id']}")
        if index == 0:
            if job["dependency"]:
                raise ValueError("standalone preparation job must not have an external dependency")
        elif job["dependency"] != f"afterok:{jobs[index - 1]['job_id']}":
            raise ValueError(f"job dependency chain drift at index {index}")
    head = subprocess.check_output(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_commit:
        raise ValueError("submission manifest commit differs from current HEAD")
    atomic_json(
        args.output,
        {
            "artifact_schema_version": 1,
            "contract_hash": args.contract_hash,
            "expected_git_commit": args.expected_commit,
            "execution_worktree": str(args.execution_worktree.resolve()),
            "jobs_in_dependency_order": jobs,
            "mode": args.mode,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"AIME_SPLIT_6K_SUBMISSION_MANIFEST mode={args.mode} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
