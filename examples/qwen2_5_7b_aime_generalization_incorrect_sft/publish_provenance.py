#!/usr/bin/env python3
"""Publish compact immutable provenance and loss history from a completed run."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import ALL_TRAINED_VARIANTS


METRIC_PATTERN = re.compile(r"step\s+(\d+):\s+(\{'train/loss'.*?\})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
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


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_metrics(log_path: Path, expected_final_step: int) -> List[Dict[str, Any]]:
    by_step: Dict[int, Dict[str, Any]] = {}
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = METRIC_PATTERN.search(line)
            if match is None:
                continue
            step = int(match.group(1))
            raw = ast.literal_eval(match.group(2))
            point = {
                "global_batch_size": int(raw["train/global_batch_size"]),
                "grad_norm": float(raw["train/grad_norm"]),
                "loss": float(raw["train/loss"]),
                "lr_pg_0": float(raw["train/lr-pg_0"]),
                "lr_pg_1": float(raw["train/lr-pg_1"]),
                "step": step,
            }
            if int(raw["train/step"]) != step:
                raise ValueError(f"metric step mismatch in {log_path}: {step}")
            previous = by_step.get(step)
            if previous is not None and previous != point:
                raise ValueError(f"conflicting duplicate metric in {log_path}: step {step}")
            by_step[step] = point
    expected_steps = list(range(expected_final_step + 1))
    if sorted(by_step) != expected_steps:
        raise ValueError(
            f"loss history is incomplete for {log_path}: "
            f"observed={len(by_step)} expected={len(expected_steps)}"
        )
    history = [by_step[step] for step in expected_steps]
    if any(point["global_batch_size"] != 64 for point in history):
        raise ValueError(f"global batch size drift in {log_path}")
    return history


def render_results(status: Dict[str, Any], summaries: List[Dict[str, Any]]) -> bytes:
    lines = [
        "# AIME distribution-generalization training results",
        "",
        f"Contract: `{status['contract_hash']}`",
        "",
        f"Training commit: `{status['repo_commit']}`",
        "",
        f"Finalized: `{status['finalized_at']}`",
        "",
        "All 12 full-parameter SFT checkpoints completed and passed the final checkpoint-manifest gate.",
        "",
        "| Variant | Job | Updates | First loss | Last loss | Change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| `{item['variant']}` | {item['job_id']} | {item['observed_update_count']} | "
            f"{item['first']['loss']:.8f} | {item['last']['loss']:.8f} | {item['loss_change']:.8f} |"
        )
    lines.extend(
        [
            "",
            "Losses are the realized training objectives. Absolute values across masking rates are not "
            "strictly comparable because each rate retains a different token set.",
            "",
            "`training_metrics.json` contains every recorded step's loss, gradient norm, learning rate, "
            "global batch size, and the SHA-256 of its source Slurm log.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


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
        "submission/stage1_parent_checkpoint_verified.json": (
            experiment_root / "manifests/stage1_parent_checkpoint_verified.json"
        ),
        "data/schedule_audit.json": experiment_root / "data/schedule_audit.json",
        "data/selection_and_tokenization_stats.json": (
            experiment_root / "data/selection_and_tokenization_stats.json"
        ),
        "data/source_artifacts.json": experiment_root / "data/source_artifacts.json",
        "data/tokenizer_control_inventory.json": experiment_root / "data/tokenizer_control_inventory.json",
        "data/variant_stats.json": experiment_root / "data/variant_stats.json",
        "handoff/checkpoint_sources.json": experiment_root / "handoff/checkpoint_sources.json",
        "handoff/comparison_sources.json": experiment_root / "handoff/comparison_sources.json",
    }
    for manifest in sorted((experiment_root / "handoff/checkpoints").glob("*.json")):
        artifacts[f"handoff/checkpoints/{manifest.name}"] = manifest
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final provenance artifacts: {missing}")

    status = load_json(artifacts["TRAINING_STATUS.json"])
    if (
        status.get("contract_hash") != args.contract_hash
        or status.get("trained_checkpoints") != 12
        or status.get("comparison_checkpoint_count") != 13
    ):
        raise ValueError("training completion status drift")
    chain = load_json(artifacts["submission/training_chain.json"])
    jobs = chain.get("jobs_in_dependency_order")
    if not isinstance(jobs, list) or len(jobs) != 16:
        raise ValueError("training chain must record exactly 16 jobs")
    jobs_by_variant = {
        str(job["variant"]): str(job["job_id"])
        for job in jobs
        if isinstance(job, dict) and job.get("variant") is not None
    }
    if tuple(jobs_by_variant) != ALL_TRAINED_VARIANTS:
        raise ValueError("training job order drift")

    metrics = []
    summaries = []
    for variant in ALL_TRAINED_VARIANTS:
        job_id = jobs_by_variant[variant]
        log_path = experiment_root / "slurm_logs" / f"q25aig-train_{job_id}.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"missing training log: {log_path}")
        expected_final_step = int(status["final_iterations"][variant])
        history = parse_metrics(log_path, expected_final_step)
        minimum = min(history, key=lambda point: point["loss"])
        maximum = max(history, key=lambda point: point["loss"])
        summary = {
            "expected_final_step": expected_final_step,
            "first": {"loss": history[0]["loss"], "step": history[0]["step"]},
            "job_id": job_id,
            "last": {"loss": history[-1]["loss"], "step": history[-1]["step"]},
            "loss_change": history[-1]["loss"] - history[0]["loss"],
            "maximum_loss": {"loss": maximum["loss"], "step": maximum["step"]},
            "minimum_loss": {"loss": minimum["loss"], "step": minimum["step"]},
            "observed_update_count": len(history),
            "source_log": str(log_path),
            "source_log_sha256": sha256(log_path),
            "variant": variant,
        }
        summaries.append(summary)
        metrics.append({**summary, "history": history})

    for relative, source in artifacts.items():
        atomic_bytes(output_root / relative, source.read_bytes())
    atomic_json(
        output_root / "training_metrics.json",
        {
            "artifact_schema_version": 1,
            "contract_hash": args.contract_hash,
            "metric_contract": "one realized train metric per optimizer update, parsed from immutable Slurm logs",
            "training_repo_commit": status["repo_commit"],
            "variants": metrics,
        },
    )
    atomic_bytes(output_root / "TRAINING_RESULTS.md", render_results(status, summaries))

    source_by_relative = {relative: str(source) for relative, source in artifacts.items()}
    source_by_relative["training_metrics.json"] = "generated from source Slurm logs named in the file"
    source_by_relative["TRAINING_RESULTS.md"] = "generated from training_metrics.json"
    published_files = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        relative = path.relative_to(output_root).as_posix()
        published_files.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": sha256(path),
                "source": source_by_relative[relative],
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
                "evaluation_jsonl": "transferred by rsync, not stored in Git",
                "runtime_token_audit_dumps": "hashes retained in TRAINING_STATUS.json",
                "selected_and_pretokenized_training_data": "excluded; hashes retained in data provenance",
            },
            "published_at": status["finalized_at"],
            "published_file_count_excluding_self": len(published_files),
            "published_files": published_files,
            "source_experiment_root": str(experiment_root),
            "training_repo_commit": status["repo_commit"],
        },
    )
    print(
        f"AIME_PROVENANCE_PUBLISHED files={len(published_files) + 1} contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
