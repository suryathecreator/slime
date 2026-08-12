#!/usr/bin/env python3
"""Verify 12 trained HF checkpoints and publish compact Hyak handoff inventories."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    ALL_TRAINED_VARIANTS,
    CHILD_VARIANTS,
    atomic_json,
    sha256_file,
)

BASE_ID = "base_qwen2_5_7b"
REMOTE_FAMILY = "checkpoints/qwen2_5_7b_aime_generalization_incorrect_sft/v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify(tool: Path, manifest: Path, checkpoint: Path) -> dict[str, Any]:
    gate = Path(__file__).with_name("hf_checkpoint_gate.py")
    subprocess.run(["python3", str(gate), "--checkpoint", str(checkpoint)], check=True)
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
    value = load_json(manifest)
    if Path(value["source_checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError(f"manifest source path differs from checkpoint: {manifest}")
    return value


def final_iteration(variant: str) -> int:
    return 187 if variant == "aime_correct_3000" else 93


def checkpoint_path(experiment_root: Path, variant: str) -> Path:
    return experiment_root / "outputs/training" / variant / "weights" / f"iter_{final_iteration(variant):07d}"


def trained_record(
    experiment_root: Path,
    variant: str,
    identity: str,
    remote_experiment: str,
) -> dict[str, Any]:
    checkpoint = checkpoint_path(experiment_root, variant)
    manifest = experiment_root / "handoff/checkpoints" / f"{variant}.json"
    checkpoint_relative = f"outputs/training/{variant}/weights/iter_{final_iteration(variant):07d}"
    manifest_relative = f"handoff/checkpoints/{variant}.json"
    return {
        "checkpoint_identity": identity,
        "id": variant,
        "kind": "new_trained",
        "model_key": "qwen2_5_7b",
        "remote_checkpoint_relative": checkpoint_relative,
        "remote_experiment_relative": remote_experiment,
        "remote_manifest_relative": manifest_relative,
        "source_checkpoint": str(checkpoint),
        "source_manifest": str(manifest),
        "variant": variant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    contract = load_json(args.contract)
    if tuple(contract["variants"]) != ALL_TRAINED_VARIANTS:
        raise ValueError("contract variant order drift")
    tool = args.repo_root / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
    handoff = root / "handoff"
    inventory_path = handoff / "checkpoint_sources.json"
    comparison_path = handoff / "comparison_sources.json"
    status_path = root / "TRAINING_STATUS.json"
    copied_base_manifest = handoff / "checkpoints/base_qwen2_5_7b.json"
    for output in (inventory_path, comparison_path, status_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite final artifact: {output}")

    required_evidence = (
        root / "data/source_artifacts.json",
        root / "data/selection_and_tokenization_stats.json",
        root / "data/variant_stats.json",
        root / "data/tokenizer_control_inventory.json",
        root / "data/schedule_audit.json",
        root / "manifests/training_chain.json",
        root / "manifests/stage1_parent_checkpoint_verified.json",
        root / "outputs/canaries/stage1/evidence/before_runtime_token_audit.json",
        root / "outputs/canaries/stage1/evidence/after_runtime_token_audit.json",
        root / "outputs/canaries/continuation_from_stage1/evidence/before_runtime_token_audit.json",
        root / "outputs/canaries/continuation_from_stage1/evidence/after_runtime_token_audit.json",
    )
    for required in required_evidence:
        if not required.is_file():
            raise FileNotFoundError(f"missing completed experiment evidence: {required}")
    evidence_hashes = {str(path.relative_to(root)): sha256_file(path) for path in required_evidence}
    chain = load_json(root / "manifests/training_chain.json")
    jobs = chain.get("jobs_in_dependency_order")
    if not isinstance(jobs, list) or len(jobs) != 16:
        raise ValueError("training chain must record exactly 16 serial jobs")
    if str(jobs[-1].get("job_id")) != str(args.job_id):
        raise ValueError("running finalizer does not match recorded chain tail")

    manifests: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {}
    for variant in ALL_TRAINED_VARIANTS:
        checkpoint = checkpoint_path(root, variant)
        manifest_path = handoff / "checkpoints" / f"{variant}.json"
        value = verify(tool, manifest_path, checkpoint)
        expected_metadata = {
            "contract_hash": args.contract_hash,
            "family": "aime_generalization",
            "final_iteration": final_iteration(variant),
            "full_sft": True,
            "variant": variant,
        }
        for key, expected in expected_metadata.items():
            if value.get(key) != expected:
                raise ValueError(f"checkpoint metadata drift: {variant}/{key}")
        manifests[variant] = value
        identities[variant] = str(value["checkpoint_manifest_sha256"])
    parent_identity = identities["aime_correct_3000"]
    for variant in CHILD_VARIANTS:
        value = manifests[variant]
        if value.get("parent_checkpoint_manifest_sha256") != parent_identity:
            raise ValueError(f"continuation parent identity drift: {variant}")
        if value.get("fresh_optimizer") is not True:
            raise ValueError(f"continuation did not record fresh optimizer: {variant}")
    parent_gate = load_json(root / "manifests/stage1_parent_checkpoint_verified.json")
    if parent_gate.get("parent_checkpoint_manifest_sha256") != parent_identity:
        raise ValueError("parent verification gate identity drift")

    base_value = verify(tool, args.base_manifest, args.base_checkpoint)
    base_identity = str(base_value["checkpoint_manifest_sha256"])
    expected_base = str(contract["base_model"]["checkpoint_identity"])
    if base_identity != expected_base:
        raise ValueError("base checkpoint identity drift")
    if copied_base_manifest.exists():
        if sha256_file(copied_base_manifest) != sha256_file(args.base_manifest):
            raise ValueError("existing copied base manifest differs from pinned source")
    else:
        copied_base_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.base_manifest, copied_base_manifest)

    remote_experiment = f"{REMOTE_FAMILY}/{args.contract_hash}"
    transfer_records = [
        trained_record(root, variant, identities[variant], remote_experiment) for variant in ALL_TRAINED_VARIANTS
    ]
    base_target = {
        "checkpoint_identity": base_identity,
        "id": BASE_ID,
        "kind": "existing_base",
        "model_key": "qwen2_5_7b",
        "remote_checkpoint_relative": "models/Qwen2.5-7B",
        "remote_experiment_relative": remote_experiment,
        "remote_manifest_relative": "handoff/checkpoints/base_qwen2_5_7b.json",
        "source_checkpoint": str(args.base_checkpoint.resolve()),
        "source_manifest": str(args.base_manifest.resolve()),
        "variant": "base",
    }
    comparison_targets = [base_target, *transfer_records]
    lightweight = [
        "TRAINING_STATUS.json",
        "config/experiment_contract.json",
        "data/schedule_audit.json",
        "data/selection_and_tokenization_stats.json",
        "data/source_artifacts.json",
        "data/tokenizer_control_inventory.json",
        "data/variant_stats.json",
        "handoff/checkpoint_sources.json",
        "handoff/comparison_sources.json",
        "manifests/training_chain.json",
    ]
    atomic_json(
        inventory_path,
        {
            "artifact_schema_version": 1,
            "checkpoint_count": len(transfer_records),
            "checkpoints": transfer_records,
            "contract_hash": args.contract_hash,
            "lightweight_handoff_files": lightweight,
            "remote_experiment_relative": remote_experiment,
            "training_datasets_transferred": False,
        },
    )
    atomic_json(
        comparison_path,
        {
            "artifact_schema_version": 1,
            "comparison_checkpoint_count": len(comparison_targets),
            "contract_hash": args.contract_hash,
            "problem_disjoint_eval_sets": ["held_in_400", "held_out_problem_400"],
            "targets": comparison_targets,
        },
    )
    commit = subprocess.check_output(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip()
    copy_diagnostics = sorted(
        str(path.relative_to(root)) for path in (root / "outputs/canaries").glob("**/*copy_diagnostic.json")
    )
    atomic_json(
        status_path,
        {
            "artifact_schema_version": 1,
            "base_checkpoint_reused": 1,
            "checkpoint_identities": {BASE_ID: base_identity, **identities},
            "comparison_checkpoint_count": len(comparison_targets),
            "contract_hash": args.contract_hash,
            "copy_regression_diagnostics_nonblocking": copy_diagnostics,
            "final_iterations": {variant: final_iteration(variant) for variant in ALL_TRAINED_VARIANTS},
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "full_parameter_sft": True,
            "repo_commit": commit,
            "required_evidence_sha256": evidence_hashes,
            "trained_checkpoints": len(transfer_records),
            "training_datasets_transferred": False,
            "transfer_checkpoint_count": len(transfer_records),
        },
    )
    print(
        "AIME_TRAINING_HANDOFF_READY "
        f"trained={len(transfer_records)} compared={len(comparison_targets)} "
        f"contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
