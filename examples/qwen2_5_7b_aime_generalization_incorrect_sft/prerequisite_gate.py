#!/usr/bin/env python3
"""Prove the external chain tail completed and resolve aged Slurm dependency use."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prerequisite evidence: {args.output}")
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            args.job_id,
            "--format=JobIDRaw,JobName,State,ExitCode,End",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 5 and fields[0] == args.job_id:
            rows.append(fields[:5])
    if len(rows) != 1:
        raise RuntimeError(f"sacct did not return one exact top-level row: {rows}")
    job_id, job_name, state, exit_code, ended_at = rows[0]
    if state != "COMPLETED" or exit_code != "0:0":
        raise RuntimeError(f"external gate is not successful: state={state} exit={exit_code}")
    control = subprocess.run(["scontrol", "show", "job", args.job_id], capture_output=True, text=True)
    controller_text = (control.stdout + control.stderr).strip()
    if control.returncode == 0:
        controller_known = True
    elif "Invalid job id" in controller_text:
        controller_known = False
    else:
        raise RuntimeError(f"could not distinguish aged Slurm job from controller error: {controller_text}")
    atomic_json(
        args.output,
        {
            "artifact_schema_version": 1,
            "controller_known": controller_known,
            "dependency_action": (
                f"attach_afterok:{job_id}" if controller_known else "omit_aged_dependency_after_sacct_success"
            ),
            "ended_at": ended_at,
            "exit_code": exit_code,
            "job_id": job_id,
            "job_name": job_name,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "sacct_exact_top_level_row": "|".join(rows[0]),
            "state": state,
        },
    )
    print(
        "AIME_EXTERNAL_GATE_SATISFIED "
        f"job={job_id} state={state} exit={exit_code} "
        f"controller_known={int(controller_known)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
