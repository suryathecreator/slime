#!/usr/bin/env python3
"""Write immutable provenance for the Qwen2.5 strict serial training chain."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument(
        "--branch",
        default="qwen2.5-7b-openr1-masked-sft-2k",
        help="Git branch whose pushed commit was submitted",
    )
    parser.add_argument("--commit", help="Explicit submitted commit; defaults to current HEAD")
    parser.add_argument("--submitted-at", help="Explicit UTC timestamp; defaults to now")
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    jobs = []
    for item in args.jobs:
        name, separator, job_id = item.partition("=")
        if separator != "=" or not name or not job_id.isdigit():
            raise ValueError(f"invalid job record: {item}")
        jobs.append({"name": name, "job_id": job_id})
    atomic_json(args.output, {
        "branch": args.branch,
        "commit": args.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "contract_hash": args.contract,
        "submitted_at": args.submitted_at or datetime.now(timezone.utc).isoformat(),
        "resource_policy": {
            "dependency": "strict linear afterok chain",
            "maximum_concurrent_gpus": 4,
            "jobs": 15,
        },
        "protected_supervision": "all added controls and literal think markup bypass masking and remain exact weight 1",
        "jobs_in_dependency_order": jobs,
    })


if __name__ == "__main__":
    main()
