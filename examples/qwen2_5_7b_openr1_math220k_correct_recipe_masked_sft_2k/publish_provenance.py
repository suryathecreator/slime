#!/usr/bin/env python3
"""Publish compact immutable provenance from a completed training root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    trace_set_sha256,
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
TRACE_FIELDS = (
    "selection_index",
    "trace_id",
    "problem_id",
    "source_row_index",
    "generation_index",
    "is_correct",
    "correctness_source",
    "source",
    "assistant_token_count",
    "assistant_token_sha256",
    "wrong_index",
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trace_manifest_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(selected):
        row = {
            "selection_index": index,
            "trace_id": str(source["trace_id"]),
            "problem_id": str(source["problem_id"]),
            "source_row_index": int(source["source_row_index"]),
            "generation_index": int(source["generation_index"]),
            "is_correct": bool(source["is_correct"]),
            "correctness_source": str(source["correctness_source"]),
            "source": source.get("source"),
            "assistant_token_count": int(source["assistant_token_count"]),
            "assistant_token_sha256": str(source["assistant_token_sha256"]),
            "wrong_index": (int(source["wrong_index"]) if source.get("wrong_index") is not None else None),
        }
        if tuple(row) != TRACE_FIELDS:
            raise AssertionError("trace manifest field order drift")
        rows.append(row)
    return rows


def validate_trace_manifest(rows: list[dict[str, Any]], expected_trace_hash: str) -> None:
    if len(rows) != 2000:
        raise ValueError(f"expected 2,000 trace identities, found {len(rows)}")
    if [row["selection_index"] for row in rows] != list(range(2000)):
        raise ValueError("selection indices are not canonical")
    trace_ids = [str(row["trace_id"]) for row in rows]
    problem_ids = [str(row["problem_id"]) for row in rows]
    if len(set(trace_ids)) != 2000 or len(set(problem_ids)) != 2000:
        raise ValueError("trace or problem identities are not unique")
    counts = Counter(bool(row["is_correct"]) for row in rows)
    if counts != Counter({True: 1000, False: 1000}):
        raise ValueError(f"class-count drift: {counts}")
    if any(not row["is_correct"] for row in rows[:1000]) or any(row["is_correct"] for row in rows[1000:]):
        raise ValueError("correct-then-wrong ordering drift")
    if [row["wrong_index"] for row in rows[1000:]] != list(range(1000)):
        raise ValueError("wrong trace indices are not canonical")
    if trace_set_sha256(trace_ids) != expected_trace_hash:
        raise ValueError("ordered trace identity hash drift")


def source_artifacts(experiment_root: Path, source_experiment_root: Path) -> dict[str, Path]:
    artifacts = {
        "TRAINING_STATUS.json": experiment_root / "TRAINING_STATUS.json",
        "submission/training_chain.json": (experiment_root / "manifests/training_chain.json"),
        "data/schedule_audit.json": experiment_root / "data/schedule_audit.json",
        "data/selection_and_tokenization_stats.json": (experiment_root / "data/selection_and_tokenization_stats.json"),
        "data/tokenizer_control_inventory.json": (experiment_root / "data/tokenizer_control_inventory.json"),
        "data/variant_stats.json": experiment_root / "data/variant_stats.json",
        "handoff/checkpoint_sources.json": (experiment_root / "handoff/checkpoint_sources.json"),
        "handoff/comparison_sources.json": (experiment_root / "handoff/comparison_sources.json"),
        "handoff/baseline_manifests/base_qwen2_5_7b.json": (
            source_experiment_root / "handoff/checkpoints/base_qwen2_5_7b.json"
        ),
        "handoff/baseline_manifests/qwen2_5_7b_8k_correct_only.json": (
            source_experiment_root / "handoff/checkpoints/qwen2_5_7b_8k.json"
        ),
    }
    for variant in VARIANTS:
        artifacts[f"handoff/checkpoints/{variant}.json"] = experiment_root / "handoff/checkpoints" / f"{variant}.json"
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source-experiment-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    source_experiment_root = args.source_experiment_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite provenance root: {output_root}")
    contract_sha256 = sha256(args.contract)
    if contract_sha256[:16] != args.contract_hash:
        raise ValueError("contract hash drift")

    contract = read_json(args.contract)
    artifacts = source_artifacts(experiment_root, source_experiment_root)
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final provenance artifacts: {missing}")
    status = read_json(artifacts["TRAINING_STATUS.json"])
    prep = read_json(artifacts["data/selection_and_tokenization_stats.json"])
    variants = read_json(artifacts["data/variant_stats.json"])
    inventory = read_json(artifacts["handoff/checkpoint_sources.json"])
    comparison = read_json(artifacts["handoff/comparison_sources.json"])
    submission = read_json(artifacts["submission/training_chain.json"])
    expected_trace_hash = str(prep["ordered_mixed_trace_ids_sha256"])
    if status["contract_hash"] != args.contract_hash:
        raise ValueError("training status contract drift")
    if status["trained_checkpoints"] != 10 or status["final_iteration"] != 124:
        raise ValueError("training completion status drift")
    if set(variants["variants"]) != set(VARIANTS) - {"unmasked"}:
        raise ValueError("variant statistics inventory drift")
    if [item["id"] for item in inventory["checkpoints"]] != list(VARIANTS):
        raise ValueError("checkpoint transfer inventory drift")
    if comparison["comparison_checkpoint_count"] != 12:
        raise ValueError("comparison inventory drift")
    if len(submission["jobs_in_dependency_order"]) != 14:
        raise ValueError("submission chain inventory drift")
    baseline_manifest_specs = {
        "base_qwen2_5_7b": (
            "handoff/baseline_manifests/base_qwen2_5_7b.json",
            "base_manifest_sha256",
        ),
        "qwen2_5_7b_8k_correct_only": (
            "handoff/baseline_manifests/qwen2_5_7b_8k_correct_only.json",
            "correct_only_manifest_sha256",
        ),
    }
    for checkpoint_id, (relative, contract_hash_key) in baseline_manifest_specs.items():
        manifest_path = artifacts[relative]
        if sha256(manifest_path) != contract["source_baselines"][contract_hash_key]:
            raise ValueError(f"source baseline manifest hash drift: {checkpoint_id}")
        manifest = read_json(manifest_path)
        if manifest["checkpoint_manifest_sha256"] != status["checkpoint_identities"][checkpoint_id]:
            raise ValueError(f"source baseline checkpoint identity drift: {checkpoint_id}")

    selected = read_jsonl(experiment_root / "data/selected_mixed_traces.jsonl")
    trace_rows = trace_manifest_rows(selected)
    validate_trace_manifest(trace_rows, expected_trace_hash)

    for relative, source in artifacts.items():
        atomic_bytes(output_root / relative, source.read_bytes())
    trace_path = output_root / "data/selected_trace_manifest.jsonl"
    trace_payload = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n" for row in trace_rows
    )
    atomic_bytes(trace_path, trace_payload)

    published_files = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        relative = path.relative_to(output_root).as_posix()
        published_files.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": sha256(path),
                "source_path": (str(artifacts[relative]) if relative in artifacts else None),
                "source_sha256": (sha256(artifacts[relative]) if relative in artifacts else None),
            }
        )
    publication = {
        "artifact_schema_version": 1,
        "contract_hash": args.contract_hash,
        "contract_relative_path": "config/experiment_contract.json",
        "contract_sha256": contract_sha256,
        "excluded_large_payloads": {
            "checkpoints_and_optimizer_state": "transferred by rsync, not stored in Git",
            "full_selected_trace_text": "excluded from Git; exact source identities and assistant token hashes are published",
            "pretokenized_training_jsonl": "excluded from Git; dataset hashes are published in the data audits",
        },
        "ordered_mixed_trace_ids_sha256": expected_trace_hash,
        "published_file_count_excluding_self": len(published_files),
        "published_files": published_files,
        "publication_time_source": "copied from finalized training status",
        "published_at": status["finalized_at"],
        "source_experiment_root": str(experiment_root),
        "trace_identity_rows": len(trace_rows),
        "training_repo_commit": status["repo_commit"],
    }
    atomic_json(output_root / "PROVENANCE_PUBLICATION.json", publication)
    print(
        "PROVENANCE_PUBLISHED "
        f"files={len(published_files) + 1} traces={len(trace_rows)} "
        f"contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
