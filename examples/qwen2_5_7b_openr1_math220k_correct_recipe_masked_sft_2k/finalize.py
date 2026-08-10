#!/usr/bin/env python3
"""Verify the ten new checkpoints and publish the Hyak comparison handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    atomic_json,
)

VARIANTS = (
    "unmasked",
    "random_mask_70",
    "inverse_tau_0p20",
    "inverse_tau_0p05",
    "random_mask_25",
    "random_mask_50",
    "random_mask_80",
    "random_mask_90",
    "margin_mask",
    "prob_ratio_mask",
)
SOURCE_REMOTE_EXPERIMENT = "checkpoints/qwen_correct_only_openr1_math220k_sft_2k/v1/3568736a3743c319"
NEW_REMOTE_FAMILY = "checkpoints/" "qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k/v1"


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


def assert_checkpoint_path(manifest: Path, checkpoint: Path, allowed_root: Path) -> None:
    value = json.loads(manifest.read_text(encoding="utf-8"))
    recorded = Path(value["source_checkpoint"]).resolve()
    checkpoint = checkpoint.resolve()
    if recorded != checkpoint:
        raise ValueError(f"checkpoint manifest path mismatch: expected={checkpoint} recorded={recorded}")
    if not checkpoint.is_relative_to(allowed_root.resolve()):
        raise ValueError(f"checkpoint escapes its allowed root: {checkpoint}")


def baseline_target(
    artifact_id: str,
    variant: str,
    checkpoint_relative: str,
    manifest_relative: str,
    identity: str,
) -> dict[str, Any]:
    return {
        "checkpoint_identity": identity,
        "id": artifact_id,
        "kind": "existing_baseline",
        "model_key": "qwen2_5_7b",
        "remote_checkpoint_relative": checkpoint_relative,
        "remote_experiment_relative": SOURCE_REMOTE_EXPERIMENT,
        "remote_manifest_relative": manifest_relative,
        "variant": variant,
    }


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
    if tuple(contract["variants"]) != VARIANTS:
        raise ValueError("variant order drift at finalization")
    tool = args.repo_root / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/" "checkpoint_manifest.py"
    handoff = experiment_root / "handoff"
    inventory_path = handoff / "checkpoint_sources.json"
    comparison_path = handoff / "comparison_sources.json"
    status_path = experiment_root / "TRAINING_STATUS.json"
    for output in (inventory_path, comparison_path, status_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite finalization artifact: {output}")

    source_specs = (
        (
            "base_qwen2_5_7b",
            "base",
            source_root / "models/Qwen2.5-7B",
            source_root / "handoff/checkpoints/base_qwen2_5_7b.json",
            contract["source_baselines"]["base_manifest_sha256"],
            contract["source_baselines"]["base_checkpoint_identity"],
            "models/Qwen2.5-7B",
            "handoff/checkpoints/base_qwen2_5_7b.json",
        ),
        (
            "qwen2_5_7b_8k_correct_only",
            "correct_only",
            source_root / "outputs/training/qwen2_5_7b_8k/correct_only/weights/iter_0000124",
            source_root / "handoff/checkpoints/qwen2_5_7b_8k.json",
            contract["source_baselines"]["correct_only_manifest_sha256"],
            contract["source_baselines"]["correct_only_checkpoint_identity"],
            "outputs/training/qwen2_5_7b_8k/correct_only/weights/iter_0000124",
            "handoff/checkpoints/qwen2_5_7b_8k.json",
        ),
    )
    comparison_targets: list[dict[str, Any]] = []
    identities: dict[str, str] = {}
    for (
        artifact_id,
        variant,
        checkpoint,
        manifest,
        manifest_hash,
        expected_identity,
        remote_checkpoint,
        remote_manifest,
    ) in source_specs:
        if sha256(manifest) != manifest_hash:
            raise ValueError(f"source baseline manifest hash drift: {manifest}")
        assert_checkpoint_path(manifest, checkpoint, source_root)
        value = verify(tool, manifest, checkpoint)
        identity = value["checkpoint_manifest_sha256"]
        if identity != expected_identity:
            raise ValueError(f"source baseline identity drift: {artifact_id}")
        identities[artifact_id] = identity
        comparison_targets.append(
            baseline_target(
                artifact_id,
                variant,
                remote_checkpoint,
                remote_manifest,
                identity,
            )
        )

    transfer_records: list[dict[str, Any]] = []
    new_remote_experiment = f"{NEW_REMOTE_FAMILY}/{args.contract_hash}"
    for variant in VARIANTS:
        checkpoint = experiment_root / "outputs" / "training" / variant / "weights" / "iter_0000124"
        manifest = handoff / "checkpoints" / f"{variant}.json"
        assert_checkpoint_path(manifest, checkpoint, experiment_root / "outputs/training" / variant)
        value = verify(tool, manifest, checkpoint)
        if value["final_iteration"] != 124 or value["full_sft"] is not True:
            raise ValueError(f"training metadata drift: {variant}")
        expected_variant = f"qwen2_5_7b_8k.{variant}"
        if value["variant"] != expected_variant:
            raise ValueError(f"checkpoint variant drift: {variant}")
        identity = value["checkpoint_manifest_sha256"]
        identities[variant] = identity
        checkpoint_relative = f"outputs/training/{variant}/weights/iter_0000124"
        manifest_relative = f"handoff/checkpoints/{variant}.json"
        transfer_records.append(
            {
                "id": variant,
                "kind": "new_trained",
                "model_key": "qwen2_5_7b",
                "remote_checkpoint_relative": checkpoint_relative,
                "remote_manifest_relative": manifest_relative,
                "source_checkpoint": str(checkpoint),
                "source_manifest": str(manifest),
                "variant": variant,
            }
        )
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

    prep_stats_path = experiment_root / "data/selection_and_tokenization_stats.json"
    variant_stats_path = experiment_root / "data/variant_stats.json"
    schedule_path = experiment_root / "data/schedule_audit.json"
    for required in (prep_stats_path, variant_stats_path, schedule_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing completed experiment evidence: {required}")
    prep_stats = json.loads(prep_stats_path.read_text(encoding="utf-8"))
    if prep_stats["invariants"]["same_order_required_for_every_variant"] is not True:
        raise ValueError("shared ordered trace invariant is not proven")

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
            "targets": comparison_targets,
        },
    )
    commit = subprocess.check_output(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip()
    atomic_json(
        status_path,
        {
            "artifact_schema_version": 1,
            "baseline_checkpoints_reused": 2,
            "checkpoint_identities": identities,
            "comparison_checkpoint_count": len(comparison_targets),
            "contract_hash": args.contract_hash,
            "final_iteration": 124,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "full_parameter_sft": True,
            "ordered_mixed_trace_ids_sha256": prep_stats["ordered_mixed_trace_ids_sha256"],
            "repo_commit": commit,
            "trained_checkpoints": len(transfer_records),
            "transfer_checkpoint_count": len(transfer_records),
        },
    )
    print(
        "TRAINING_HANDOFF_READY "
        f"new_checkpoints={len(transfer_records)} "
        f"comparison_checkpoints={len(comparison_targets)} "
        f"contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
