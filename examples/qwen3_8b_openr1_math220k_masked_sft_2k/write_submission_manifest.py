#!/usr/bin/env python3
"""Write provenance for the serialized training-only completion chain."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    atomic_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract-40k", required=True)
    parser.add_argument("--contract-2k", required=True)
    parser.add_argument(
        "--reused-checkpoint",
        action="append",
        default=[],
        help="NAME=PATH checkpoint preserved instead of resubmitted",
    )
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    ordered_jobs: list[dict[str, str]] = []
    for item in args.jobs:
        name, separator, job_id = item.partition("=")
        if separator != "=" or not name or not job_id.isdigit():
            raise ValueError(f"invalid job record: {item}")
        ordered_jobs.append({"name": name, "job_id": job_id})
    reused_checkpoints: list[dict[str, str]] = []
    for item in args.reused_checkpoint:
        name, separator, path = item.partition("=")
        if separator != "=" or not name or not path.startswith("/"):
            raise ValueError(f"invalid reused checkpoint record: {item}")
        reused_checkpoints.append({"name": name, "path": path})
    atomic_json(
        args.output,
        {
            "branch": "qwen3-8b-openr1-masked-sft-40k",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "contracts": {
                "recovery_40k": args.contract_40k,
                "suite_2k": args.contract_2k,
            },
            "completion_note": "Completes the already-started Axolotl-masked-SFT work after repairing fractional loss accounting and the MATH-500 path.",
            "resource_policy": {
                "dependency": "strict linear afterok chain",
                "maximum_concurrent_gpus": 4,
                "launch_scope": "data preparation, margin scoring, and full-SFT training only",
                "order": "all 2K jobs first, then remaining 40K training",
            },
            "evaluation_policy": {
                "execution": "deferred_to_another_system",
                "dataset": "MATH-500",
                "decoding": "greedy",
                "dynamic_response_budget": "32768 - rendered_prompt_tokens - 64",
                "scorer": "math500_strict_boxed_scorer_v3",
                "stop_token_ids": [151643, 151645],
            },
            "reused_completed_checkpoints": reused_checkpoints,
            "jobs_in_dependency_order": ordered_jobs,
        },
    )


if __name__ == "__main__":
    main()
