#!/usr/bin/env python3
"""Publish compact immutable provenance from a completed AIME split-control run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.data_utils import (
    DATASET_ROWS,
    EPOCHS,
    GLOBAL_BATCH_SIZE,
    UPDATES,
    VARIANTS,
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
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    atomic_bytes(path, payload)


def validate_metrics(metrics: dict[str, Any]) -> None:
    updates = metrics.get("optimizer_updates_per_variant")
    if updates != UPDATES:
        raise ValueError(f"expected {UPDATES} optimizer updates per variant, found {updates!r}")
    variants = metrics.get("variants")
    if not isinstance(variants, list) or [item.get("variant") for item in variants] != list(VARIANTS):
        raise ValueError("training metrics variant order drift")
    for item in variants:
        history = item.get("history")
        if not isinstance(history, list) or len(history) != updates:
            raise ValueError(f"incomplete loss history for {item['variant']}")
        if [record.get("step") for record in history] != list(range(updates)):
            raise ValueError(f"non-contiguous loss history for {item['variant']}")
        if item.get("first") != history[0] or item.get("last") != history[-1]:
            raise ValueError(f"first/last loss summary drift for {item['variant']}")
        if item.get("loss_change") != history[-1]["loss"] - history[0]["loss"]:
            raise ValueError(f"loss change summary drift for {item['variant']}")
        if {record["global_batch_size"] for record in history} != {GLOBAL_BATCH_SIZE}:
            raise ValueError(f"global batch size drift for {item['variant']}")


def render_training_summary(status: dict[str, Any], metrics: dict[str, Any]) -> str:
    updates = int(metrics["optimizer_updates_per_variant"])
    if set(status["final_iterations"].values()) != {updates - 1}:
        raise ValueError("status final iterations do not match metric history")
    rows = []
    for item in metrics["variants"]:
        history = item["history"]
        losses = [record["loss"] for record in history]
        rows.append(
            "| {variant} | {job} | {first:.6f} | {last:.6f} | {change:.6f} | {minimum:.6f} | {step} | "
            "{first_five:.6f} | {last_five:.6f} | {grad:.6f} |".format(
                variant=item["variant"],
                job=item["job_id"],
                first=losses[0],
                last=losses[-1],
                change=losses[-1] - losses[0],
                minimum=min(losses),
                step=losses.index(min(losses)),
                first_five=statistics.mean(losses[:5]),
                last_five=statistics.mean(losses[-5:]),
                grad=history[-1]["grad_norm"],
            )
        )
    return "\n".join(
        [
            "# Completed training provenance",
            "",
            f"- Contract: `{status['contract_hash']}`",
            f"- Training commit: `{status['repo_commit']}`",
            f"- Finalized: `{status['finalized_at']}`",
            f"- Four independent full-parameter Qwen2.5-7B SFT runs completed at iteration {updates - 1}.",
            f"- Each variant trained {EPOCHS} epochs over {DATASET_ROWS} rows at global batch "
            f"{GLOBAL_BATCH_SIZE}, giving {updates} optimizer updates.",
            f"- The complete {updates}-update metric history for every variant is in "
            "`handoff/training_metrics.json`.",
            "",
            "| Variant | Job | First loss | Last loss | Change | Minimum | Min step | First-5 avg | "
            "Last-5 avg | Final grad norm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The two correct conditions converge to a visibly lower loss than the two incorrect "
            "conditions. That is a fit statistic on different target distributions, not evidence "
            "about generalization; the held-in/held-out comparison is decided by the Hyak "
            "evaluation described in `handoff/EVAL_HANDOFF.md`.",
            "",
            "This Git publication intentionally excludes checkpoints, optimizer/full state, "
            "selected and expanded training data, raw source generations, and canary payloads. "
            "Checkpoints, the two 106-problem evaluation sets, and the compact handoff are "
            "transferred separately with `rsync_to_klone.sh`.",
            "",
        ]
    )


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
        "handoff/EVAL_HANDOFF.md": experiment_root / "handoff/EVAL_HANDOFF.md",
        "handoff/checkpoint_sources.json": experiment_root / "handoff/checkpoint_sources.json",
        "handoff/comparison_sources.json": experiment_root / "handoff/comparison_sources.json",
        "handoff/eval_contract.json": experiment_root / "handoff/eval_contract.json",
        "handoff/training_metrics.json": experiment_root / "handoff/training_metrics.json",
        "handoff/checkpoints/base_qwen2_5_7b.json": experiment_root / "handoff/checkpoints/base_qwen2_5_7b.json",
    }
    for variant in VARIANTS:
        artifacts[f"handoff/checkpoints/{variant}.json"] = experiment_root / "handoff/checkpoints" / f"{variant}.json"
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final provenance artifacts: {missing}")

    status = json.loads(artifacts["TRAINING_STATUS.json"].read_text(encoding="utf-8"))
    metrics = json.loads(artifacts["handoff/training_metrics.json"].read_text(encoding="utf-8"))
    validate_metrics(metrics)
    expected_iterations = {variant: UPDATES - 1 for variant in VARIANTS}
    if (
        status.get("contract_hash") != args.contract_hash
        or status.get("trained_checkpoints") != len(VARIANTS)
        or status.get("comparison_checkpoint_count") != len(VARIANTS) + 1
        or status.get("full_parameter_sft") is not True
        or status.get("final_iterations") != expected_iterations
    ):
        raise ValueError("training completion status drift")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for relative, source in artifacts.items():
            atomic_bytes(staging / relative, source.read_bytes())
        atomic_bytes(staging / "TRAINING_SUMMARY.md", render_training_summary(status, metrics).encode("utf-8"))

        published_files = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            source = artifacts.get(relative)
            published_files.append(
                {
                    "bytes": path.stat().st_size,
                    "origin": "copied" if source is not None else "generated",
                    "path": relative,
                    "sha256": sha256(path),
                    "source_relative": str(source.relative_to(experiment_root)) if source is not None else None,
                    "source_sha256": sha256(source) if source is not None else None,
                }
            )
        atomic_json(
            staging / "PROVENANCE_PUBLICATION.json",
            {
                "artifact_schema_version": 1,
                "contract_hash": args.contract_hash,
                "contract_relative_path": "config/experiment_contract.json",
                "contract_sha256": contract_sha256,
                "excluded_large_payloads": {
                    "canary_payloads": "excluded from Git; required audit hashes are recorded in TRAINING_STATUS.json",
                    "checkpoints_and_optimizer_state": "transferred by rsync, not stored in Git",
                    "evaluation_sets": "held_in_106.jsonl and held_out_106.jsonl are transferred by rsync",
                    "raw_source_generations": "excluded from Git; pinned dataset revision and hashes are recorded",
                    "training_datasets": "excluded from Git; selection and tokenization hashes are recorded",
                },
                "published_at": status["finalized_at"],
                "published_file_count_excluding_self": len(published_files),
                "published_files": published_files,
                "source_experiment_root": str(experiment_root),
                "training_repo_commit": status["repo_commit"],
            },
        )
        staging.replace(output_root)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

    print(
        f"AIME_SPLIT_6K_PROVENANCE_PUBLISHED files={len(published_files) + 1} contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
