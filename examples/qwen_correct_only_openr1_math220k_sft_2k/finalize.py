#!/usr/bin/env python3
"""Verify all nine HF artifacts and publish the manual Hyak handoff inventory."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import atomic_json
from examples.qwen_correct_only_openr1_math220k_sft_2k.experiment import (
    MODEL_ORDER,
    RUN_ORDER,
    load_contract,
)


def verify(tool: Path, manifest: Path, checkpoint: Path) -> dict[str, Any]:
    subprocess.run(
        [
            "python3",
            str(tool),
            "verify",
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
        ],
        check=True,
    )
    return json.loads(manifest.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract()
    tool = (
        args.repo_root
        / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
    )
    handoff = args.experiment_root / "handoff"
    inventory_path = handoff / "checkpoint_sources.json"
    status_path = args.experiment_root / "TRAINING_STATUS.json"
    for output in (inventory_path, status_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite finalization artifact: {output}")

    records: list[dict[str, Any]] = []
    identities: dict[str, str] = {}
    for model_key in MODEL_ORDER:
        model = contract["models"][model_key]
        checkpoint = args.experiment_root / "models" / model["hf_dir_name"]
        manifest = handoff / "checkpoints" / f"base_{model_key}.json"
        value = verify(tool, manifest, checkpoint)
        artifact_id = f"base_{model_key}"
        identities[artifact_id] = value["checkpoint_manifest_sha256"]
        records.append(
            {
                "id": artifact_id,
                "kind": "base",
                "model_key": model_key,
                "run_key": None,
                "source_checkpoint": str(checkpoint),
                "source_manifest": str(manifest),
                "remote_checkpoint_relative": f"models/{model['hf_dir_name']}",
                "remote_manifest_relative": f"handoff/checkpoints/{artifact_id}.json",
                "variant": "base",
            }
        )
    copy_diagnostics: dict[str, Any] = {}
    for run_key in RUN_ORDER:
        checkpoint = (
            args.experiment_root
            / "outputs"
            / "training"
            / run_key
            / "correct_only"
            / "weights"
            / "iter_0000124"
        )
        manifest = handoff / "checkpoints" / f"{run_key}.json"
        value = verify(tool, manifest, checkpoint)
        identities[run_key] = value["checkpoint_manifest_sha256"]
        records.append(
            {
                "id": run_key,
                "kind": "trained",
                "model_key": contract["runs"][run_key]["model"],
                "run_key": run_key,
                "source_checkpoint": str(checkpoint),
                "source_manifest": str(manifest),
                "remote_checkpoint_relative": (
                    f"outputs/training/{run_key}/correct_only/weights/iter_0000124"
                ),
                "remote_manifest_relative": f"handoff/checkpoints/{run_key}.json",
                "variant": "correct_only",
            }
        )
        schedule = args.experiment_root / "data" / "schedule_audits" / f"{run_key}.json"
        audit = args.experiment_root / "outputs" / "canaries" / run_key / "runtime_batch_audit.json"
        generation = (
            args.experiment_root
            / "outputs"
            / "canaries"
            / run_key
            / "generation_before_after.json"
        )
        for required in (schedule, audit, generation):
            if not required.is_file():
                raise FileNotFoundError(f"missing completed run evidence: {required}")
        generation_value = json.loads(generation.read_text(encoding="utf-8"))
        copy_diagnostics[run_key] = generation_value["copy_regression"]

    selection_stats = json.loads(
        (args.experiment_root / "data" / "selection_and_tokenization_stats.json").read_text(
            encoding="utf-8"
        )
    )
    if selection_stats["invariants"]["same_ordered_2k_across_all_8k_models"] is not True:
        raise ValueError("shared 8K selection invariant is not proven")
    inventory = {
        "artifact_schema_version": 1,
        "checkpoint_count": len(records),
        "checkpoints": records,
        "contract_hash": args.contract_hash,
        "remote_experiment_relative": (
            f"checkpoints/qwen_correct_only_openr1_math220k_sft_2k/v1/{args.contract_hash}"
        ),
    }
    atomic_json(inventory_path, inventory)
    commit = subprocess.check_output(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    atomic_json(
        status_path,
        {
            "artifact_schema_version": 1,
            "base_checkpoints": 4,
            "canary_copy_regression_policy": "diagnostic_only_does_not_block_training",
            "canary_copy_diagnostics": copy_diagnostics,
            "checkpoint_identities": identities,
            "contract_hash": args.contract_hash,
            "final_iteration": 124,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "full_parameter_sft": True,
            "repo_commit": commit,
            "shared_8k_ordered_trace_sha256": selection_stats["selection"]["8k_shared"][
                "ordered_trace_sha256"
            ],
            "trained_checkpoints": 5,
            "verified_checkpoint_count": len(records),
        },
    )
    print(
        f"TRAINING_HANDOFF_READY checkpoints={len(records)} contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
