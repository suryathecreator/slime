#!/usr/bin/env python3
"""Write immutable provenance for the strict serial training-only chain."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    jobs = []
    for item in args.jobs:
        name, separator, job_id = item.partition("=")
        if separator != "=" or not name or not job_id.isdigit():
            raise ValueError(f"invalid job record: {item}")
        jobs.append({"name": name, "job_id": job_id})
    atomic_json(
        args.output,
        {
            "branch": args.branch,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "contract_hash": args.contract,
            "jobs_in_dependency_order": jobs,
            "resource_policy": {
                "dependency": "strict linear afterok chain",
                "jobs": len(jobs),
                "maximum_concurrent_gpus": 4,
                "submits_evaluation": False,
                "submits_transfer": False,
            },
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
