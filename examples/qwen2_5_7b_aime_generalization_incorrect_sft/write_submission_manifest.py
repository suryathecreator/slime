#!/usr/bin/env python3
"""Write an atomic dry-run, in-progress, or final Slurm chain manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--upstream-gate", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry_run", "in_progress", "submitted"), required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        raise FileExistsError(f"refusing to overwrite chain manifest: {args.output}")
    jobs = []
    with args.jobs_tsv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            jobs.append(
                {
                    "dependency": row["dependency"] or None,
                    "job_id": row["job_id"] or None,
                    "name": row["name"],
                    "script": row["script"],
                    "variant": row["variant"] or None,
                }
            )
    if args.mode in {"dry_run", "submitted"} and len(jobs) != 16:
        raise ValueError(f"{args.mode} manifest requires 16 jobs, got {len(jobs)}")
    if args.mode == "submitted":
        ids = [str(row["job_id"]) for row in jobs]
        if any(not value.isdigit() for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("submitted job IDs must be unique positive integers")
        for previous, current in zip(jobs, jobs[1:], strict=True):
            if current["dependency"] != f"afterok:{previous['job_id']}":
                raise ValueError("submitted chain is not strictly serial")
    gate = json.loads(args.upstream_gate.read_text(encoding="utf-8"))
    expected_first = f"afterok:{gate['job_id']}" if gate["controller_known"] else None
    if jobs and jobs[0]["dependency"] != expected_first:
        raise ValueError("first-job dependency differs from resolved external gate")
    commit = subprocess.check_output(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != args.expected_commit:
        raise ValueError(f"repository changed during submission: expected={args.expected_commit} actual={commit}")
    branch = subprocess.check_output(["git", "-C", str(args.repo_root), "branch", "--show-current"], text=True).strip()
    atomic_json(
        args.output,
        {
            "artifact_schema_version": 1,
            "branch": branch,
            "commit": commit,
            "contract_hash": args.contract_hash,
            "external_prerequisite": gate,
            "expected_runtime_commit": args.expected_commit,
            "jobs_in_dependency_order": jobs,
            "mode": args.mode,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "resource_policy": {
                "full_training_walltime": "04:00:00",
                "jobs": 16,
                "maximum_concurrent_gpus": 4,
                "other_job_walltime": "02:00:00",
                "strict_linear_afterok": True,
                "submits_evaluation": False,
                "submits_transfer": False,
            },
        },
    )
    print(
        f"AIME_CHAIN_MANIFEST_WRITTEN mode={args.mode} jobs={len(jobs)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
