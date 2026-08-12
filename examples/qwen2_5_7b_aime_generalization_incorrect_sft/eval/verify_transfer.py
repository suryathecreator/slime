#!/usr/bin/env python3
"""Read-only validation for the checkpoint-only Hyak handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.aime_eval import (
    TARGETS,
    normalize_rows,
    sha256_file,
)
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.hf_checkpoint_gate import (
    validate_checkpoint,
)

TRANSFER_TARGETS = TARGETS[1:]
FORBIDDEN_COMPONENTS = frozenset(
    {
        "datasets",
        "optimizer",
        "optim",
        "torch_dist",
        "selected",
    }
)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def safe_relative(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or any(part.casefold() in FORBIDDEN_COMPONENTS for part in path.parts):
        raise RuntimeError(f"unsafe {label}: {value}")
    return path


def validate_hf_checkpoint(checkpoint: Path, manifest_path: Path, artifact_id: str, expected_identity: str) -> None:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing finalized checkpoint artifact: {artifact_id}")
    validate_checkpoint(checkpoint)
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    actual = {path.name for path in checkpoint.iterdir() if path.is_file()}
    missing = sorted(required - actual)
    if missing:
        raise RuntimeError(f"incomplete final HF checkpoint {artifact_id}: {missing}")
    for path in checkpoint.rglob("*"):
        relative = path.relative_to(checkpoint)
        lowered = [part.casefold() for part in relative.parts]
        if path.is_symlink() or any(
            part in FORBIDDEN_COMPONENTS or part.startswith("optim") or "optimizer" in part or "torch_dist" in part
            for part in lowered
        ):
            raise RuntimeError(f"non-HF training state in {artifact_id}: {relative}")
    manifest = load_json(manifest_path)
    if manifest.get("checkpoint_manifest_sha256") != expected_identity:
        raise RuntimeError(f"checkpoint identity mismatch in manifest: {artifact_id}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"checkpoint manifest has no file inventory: {artifact_id}")
    expected_files = {str(item["name"]): int(item["bytes"]) for item in files}
    actual_files = {
        path.relative_to(checkpoint).as_posix(): path.stat().st_size
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError(f"checkpoint size inventory mismatch: {artifact_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-in", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, required=True)
    parser.add_argument("--checkpoint-inventory", type=Path, required=True)
    parser.add_argument("--comparison-inventory", type=Path, required=True)
    args = parser.parse_args()

    held_in = normalize_rows(args.held_in, "held_in")
    held_out = normalize_rows(args.held_out, "held_out_problem")
    overlap = {row["problem_id"] for row in held_in} & {row["problem_id"] for row in held_out}
    if overlap:
        raise RuntimeError(f"eval split problem overlap: {sorted(overlap)[:10]}")

    checkpoints = load_json(args.checkpoint_inventory)
    records = checkpoints.get("checkpoints")
    if (
        checkpoints.get("checkpoint_count") != len(TRANSFER_TARGETS)
        or checkpoints.get("training_datasets_transferred") is not False
        or not isinstance(records, list)
        or tuple(row.get("id") for row in records) != TRANSFER_TARGETS
    ):
        raise RuntimeError("checkpoint-only transfer inventory must contain the 12 trained targets")
    for row in records:
        safe_relative(str(row["remote_checkpoint_relative"]), "checkpoint destination")
        safe_relative(str(row["remote_manifest_relative"]), "manifest destination")
        if not Path(row["source_manifest"]).is_file():
            raise FileNotFoundError(f"missing finalized checkpoint artifact: {row['id']}")
        validate_hf_checkpoint(
            Path(row["source_checkpoint"]),
            Path(row["source_manifest"]),
            str(row["id"]),
            str(row["checkpoint_identity"]),
        )

    comparison = load_json(args.comparison_inventory)
    targets = comparison.get("targets")
    if (
        comparison.get("comparison_checkpoint_count") != len(TARGETS)
        or not isinstance(targets, list)
        or tuple(row.get("id") for row in targets) != TARGETS
    ):
        raise RuntimeError("comparison inventory must contain all 13 targets in canonical order")
    for row in targets:
        safe_relative(str(row["remote_checkpoint_relative"]), "comparison checkpoint")
        safe_relative(str(row["remote_manifest_relative"]), "comparison manifest")
    transferred = {row["id"]: row for row in records}
    for row in targets[1:]:
        source = transferred[row["id"]]
        for field in (
            "checkpoint_identity",
            "remote_checkpoint_relative",
            "remote_manifest_relative",
            "source_checkpoint",
            "source_manifest",
        ):
            if row.get(field) != source.get(field):
                raise RuntimeError(f"comparison/transfer inventory mismatch: {row['id']}/{field}")

    print(
        json.dumps(
            {
                "checkpoint_count": len(records),
                "held_in_rows": len(held_in),
                "held_in_sha256": sha256_file(args.held_in),
                "held_out_rows": len(held_out),
                "held_out_sha256": sha256_file(args.held_out),
                "status": "valid",
                "training_datasets_transferred": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
