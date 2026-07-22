#!/usr/bin/env python3
"""Write the compact Slurm-chain provenance manifest."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import atomic_json


def main() -> None:
    """Write a complete submission manifest from name=job arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    jobs = dict(item.split("=", 1) for item in args.jobs)
    if any(not value.isdigit() for value in jobs.values()):
        raise ValueError(f"invalid Slurm job IDs: {jobs}")
    atomic_json(
        args.output,
        {
            "branch": "qwen3-8b-openr1-masked-sft-40k",
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "contract_hash": args.contract_hash,
            "jobs": jobs,
            "serial_semantics": "prepare -> margins -> three independent full-SFT variants -> three MATH-500 evals -> report",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
