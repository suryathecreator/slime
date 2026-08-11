#!/usr/bin/env python3
"""Verify six new checkpoints and publish the 18-target comparison handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k.build_random_masks import (
    EXPECTED_VARIANTS,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    atomic_json,
)

NEW_REMOTE_FAMILY = "checkpoints/" "qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def assert_checkpoint_path(manifest: Path, checkpoint: Path, root: Path) -> None:
    value = json.loads(manifest.read_text(encoding="utf-8"))
    recorded = Path(value["source_checkpoint"]).resolve()
    checkpoint = checkpoint.resolve()
    if recorded != checkpoint:
        raise ValueError(f"checkpoint path mismatch: expected={checkpoint} recorded={recorded}")
    if not checkpoint.is_relative_to(root.resolve()):
        raise ValueError(f"checkpoint escapes allowed root: {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source-experiment-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    source_root = args.source_experiment_root.resolve()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    variants = tuple(item["id"] for item in contract["variants"])
    if variants != EXPECTED_VARIANTS:
        raise ValueError("variant order drift at finalization")
    source_contract = contract["source_experiment"]
    source_status_path = source_root / "TRAINING_STATUS.json"
    source_comparison_path = source_root / "handoff/comparison_sources.json"
    if sha256(source_status_path) != source_contract["training_status_sha256"]:
        raise ValueError("source training status hash drift")
    if sha256(source_comparison_path) != source_contract["comparison_sources_sha256"]:
        raise ValueError("source comparison inventory hash drift")
    source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
    source_comparison = json.loads(source_comparison_path.read_text(encoding="utf-8"))
    if source_comparison["comparison_checkpoint_count"] != 12:
        raise ValueError("source comparison target-count drift")
    comparison_targets = list(source_comparison["targets"])
    if len({target["id"] for target in comparison_targets}) != 12:
        raise ValueError("duplicate source comparison target")
    identities = {str(target["id"]): str(target["checkpoint_identity"]) for target in comparison_targets}
    if identities != source_status["checkpoint_identities"]:
        raise ValueError("source checkpoint identity inventory drift")

    tool = args.repo_root / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/" "checkpoint_manifest.py"
    handoff = experiment_root / "handoff"
    inventory_path = handoff / "checkpoint_sources.json"
    comparison_path = handoff / "comparison_sources.json"
    status_path = experiment_root / "TRAINING_STATUS.json"
    for output in (inventory_path, comparison_path, status_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite final artifact: {output}")

    transfer_records: list[dict[str, Any]] = []
    new_remote_experiment = f"{NEW_REMOTE_FAMILY}/{args.contract_hash}"
    for variant in variants:
        checkpoint = experiment_root / "outputs/training" / variant / "weights/iter_0000124"
        manifest = handoff / "checkpoints" / f"{variant}.json"
        assert_checkpoint_path(
            manifest,
            checkpoint,
            experiment_root / "outputs/training" / variant,
        )
        value = verify(tool, manifest, checkpoint)
        if value["final_iteration"] != 124 or value["full_sft"] is not True:
            raise ValueError(f"training metadata drift: {variant}")
        if value["variant"] != f"qwen2_5_7b_8k.{variant}":
            raise ValueError(f"checkpoint variant drift: {variant}")
        identity = str(value["checkpoint_manifest_sha256"])
        identities[variant] = identity
        checkpoint_relative = f"outputs/training/{variant}/weights/iter_0000124"
        manifest_relative = f"handoff/checkpoints/{variant}.json"
        record = {
            "id": variant,
            "kind": "new_trained",
            "model_key": "qwen2_5_7b",
            "remote_checkpoint_relative": checkpoint_relative,
            "remote_manifest_relative": manifest_relative,
            "source_checkpoint": str(checkpoint),
            "source_manifest": str(manifest),
            "variant": variant,
        }
        transfer_records.append(record)
        comparison_targets.append(
            {
                "checkpoint_identity": identity,
                "id": variant,
                "kind": "new_trained",
                "model_key": "qwen2_5_7b",
                "remote_checkpoint_relative": checkpoint_relative,
                "remote_experiment_relative": new_remote_experiment,
                "remote_manifest_relative": manifest_relative,
                "variant": variant,
            }
        )

    if len(transfer_records) != 6 or len(comparison_targets) != 18:
        raise ValueError("final handoff count drift")
    if len({target["id"] for target in comparison_targets}) != 18:
        raise ValueError("duplicate final comparison target")
    prep_path = experiment_root / "data/selection_and_tokenization_stats.json"
    stats_path = experiment_root / "data/variant_stats.json"
    schedule_path = experiment_root / "data/schedule_audit.json"
    source_artifacts_path = experiment_root / "data/source_artifacts.json"
    for required in (prep_path, stats_path, schedule_path, source_artifacts_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing experiment evidence: {required}")
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    expected_trace_hash = source_contract["ordered_mixed_trace_ids_sha256"]
    if prep["ordered_mixed_trace_ids_sha256"] != expected_trace_hash:
        raise ValueError("source trace identity hash drift")
    if stats["ordered_mixed_trace_ids_sha256"] != expected_trace_hash:
        raise ValueError("variant trace identity hash drift")

    atomic_json(
        inventory_path,
        {
            "artifact_schema_version": 1,
            "checkpoint_count": len(transfer_records),
            "checkpoints": transfer_records,
            "contract_hash": args.contract_hash,
            "lightweight_handoff_files": [
                "TRAINING_STATUS.json",
                "config/experiment_contract.json",
                "data/schedule_audit.json",
                "data/selection_and_tokenization_stats.json",
                "data/source_artifacts.json",
                "data/tokenizer_control_inventory.json",
                "data/variant_stats.json",
                "handoff/checkpoint_sources.json",
                "handoff/comparison_sources.json",
            ],
            "remote_experiment_relative": new_remote_experiment,
        },
    )
    atomic_json(
        comparison_path,
        {
            "artifact_schema_version": 1,
            "comparison_checkpoint_count": len(comparison_targets),
            "contract_hash": args.contract_hash,
            "same_ordered_mixed_trace_set_for_new_variants": True,
            "source_comparison_checkpoint_count": 12,
            "source_comparison_sha256": source_contract["comparison_sources_sha256"],
            "targets": comparison_targets,
        },
    )
    commit = subprocess.check_output(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip()
    atomic_json(
        status_path,
        {
            "artifact_schema_version": 1,
            "checkpoint_identities": identities,
            "comparison_checkpoint_count": len(comparison_targets),
            "contract_hash": args.contract_hash,
            "final_iteration": 124,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "full_parameter_sft": True,
            "mask_probabilities": [0.05, 0.15],
            "mask_seeds": [42, 43, 44],
            "ordered_mixed_trace_ids_sha256": expected_trace_hash,
            "repo_commit": commit,
            "source_comparison_checkpoints_reused": 12,
            "source_contract_hash": source_contract["contract_hash"],
            "trained_checkpoints": len(transfer_records),
            "transfer_checkpoint_count": len(transfer_records),
        },
    )
    print(
        "REPLICATE_TRAINING_HANDOFF_READY "
        f"new_checkpoints={len(transfer_records)} "
        f"comparison_checkpoints={len(comparison_targets)} "
        f"contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
