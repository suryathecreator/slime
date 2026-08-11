#!/usr/bin/env python3
"""Publish compact immutable provenance from a completed replicate run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k.build_random_masks import (
    EXPECTED_VARIANTS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    atomic_bytes(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite provenance root: {output_root}")
    contract_sha256 = sha256(args.contract)
    if contract_sha256[:16] != args.contract_hash:
        raise ValueError("contract hash drift")
    artifacts = {
        "TRAINING_STATUS.json": experiment_root / "TRAINING_STATUS.json",
        "submission/training_chain.json": experiment_root / "manifests/training_chain.json",
        "data/schedule_audit.json": experiment_root / "data/schedule_audit.json",
        "data/selection_and_tokenization_stats.json": experiment_root / "data/selection_and_tokenization_stats.json",
        "data/source_artifacts.json": experiment_root / "data/source_artifacts.json",
        "data/tokenizer_control_inventory.json": experiment_root / "data/tokenizer_control_inventory.json",
        "data/variant_stats.json": experiment_root / "data/variant_stats.json",
        "handoff/checkpoint_sources.json": experiment_root / "handoff/checkpoint_sources.json",
        "handoff/comparison_sources.json": experiment_root / "handoff/comparison_sources.json",
    }
    for variant in EXPECTED_VARIANTS:
        artifacts[f"handoff/checkpoints/{variant}.json"] = experiment_root / "handoff/checkpoints" / f"{variant}.json"
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final provenance artifacts: {missing}")
    status = json.loads(artifacts["TRAINING_STATUS.json"].read_text(encoding="utf-8"))
    if (
        status["contract_hash"] != args.contract_hash
        or status["trained_checkpoints"] != 6
        or status["comparison_checkpoint_count"] != 18
        or status["final_iteration"] != 124
    ):
        raise ValueError("training completion status drift")
    for relative, source in artifacts.items():
        atomic_bytes(output_root / relative, source.read_bytes())
    published_files = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        relative = path.relative_to(output_root).as_posix()
        published_files.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": sha256(path),
                "source_path": str(artifacts[relative]),
                "source_sha256": sha256(artifacts[relative]),
            }
        )
    atomic_json(
        output_root / "PROVENANCE_PUBLICATION.json",
        {
            "artifact_schema_version": 1,
            "contract_hash": args.contract_hash,
            "contract_relative_path": "config/experiment_contract.json",
            "contract_sha256": contract_sha256,
            "excluded_large_payloads": {
                "checkpoints_and_optimizer_state": "transferred by rsync, not stored in Git",
                "pretokenized_training_jsonl": "excluded from Git; hashes are recorded in variant_stats.json",
                "source_unmasked_dataset": "already pinned and published by the source experiment",
            },
            "ordered_mixed_trace_ids_sha256": status["ordered_mixed_trace_ids_sha256"],
            "published_at": status["finalized_at"],
            "published_file_count_excluding_self": len(published_files),
            "published_files": published_files,
            "source_experiment_root": str(experiment_root),
            "training_repo_commit": status["repo_commit"],
        },
    )
    print(
        f"REPLICATE_PROVENANCE_PUBLISHED files={len(published_files) + 1} " f"contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
