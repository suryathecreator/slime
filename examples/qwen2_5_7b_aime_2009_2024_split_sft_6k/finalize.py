#!/usr/bin/env python3
"""Verify four checkpoints and publish the compact Hyak evaluation handoff."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.data_utils import (
    VARIANTS,
    atomic_json,
    read_jsonl,
    sha256_file,
)

BASE_ID = "base_qwen2_5_7b"
REMOTE_FAMILY = "checkpoints/qwen2_5_7b_aime_2009_2024_split_sft_6k/v1"
METRIC_PATTERN = re.compile(r"step\s+(\d+):\s+(\{'train/loss'.*?\})")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def checkpoint_path(root: Path, variant: str, final_iteration: int) -> Path:
    return root / "outputs/training" / variant / f"weights/iter_{final_iteration:07d}"


def verify_checkpoint(*, tool: Path, gate: Path, manifest: Path, checkpoint: Path) -> dict[str, Any]:
    subprocess.run([sys.executable, str(gate), "--checkpoint", str(checkpoint)], check=True)
    subprocess.run(
        [
            sys.executable,
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
        raise ValueError(f"manifest/checkpoint path drift: {manifest}")
    return value


def parse_metrics(log_path: Path, final_iteration: int) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
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
                raise ValueError(f"metric step mismatch: {log_path}/{step}")
            # A requeue may legitimately rerun an update that logged before its
            # durable full-state checkpoint. The final occurrence is the one
            # belonging to the surviving resumed trajectory.
            by_step[step] = point
    if sorted(by_step) != list(range(final_iteration + 1)):
        raise ValueError(f"loss history is incomplete for {log_path}: observed={sorted(by_step)}")
    history = [by_step[step] for step in range(final_iteration + 1)]
    if any(point["global_batch_size"] != 64 for point in history):
        raise ValueError(f"global batch size drift in {log_path}")
    return history


def render_eval_handoff() -> str:
    return """# Hyak evaluation handoff

Evaluate the verified Qwen2.5-7B base and all four trained checkpoints on both
transferred 240-problem sets. The files are `held_in_240.jsonl` and
`held_out_240.jsonl`; together they are a disjoint, exhaustive partition of the
480 AIME 2009–2024 problems.

Use each row's exact `rendered_prompt` bytes directly. It has exactly one user
turn and no system turn:

```text
<|im_start|>user
Please reason step by step, and put your final answer within \\boxed{}.

Problem:
{problem}<|im_end|>
<|im_start|>assistant
```

Do not call the Qwen2.5 default chat template, because it would insert `You are
a helpful assistant.` and would differ from both source generation and
training. Do not insert a stored response.

Use `aime_last_boxed_integer_scorer_v1` from SLIME. Score every generated
completion, including cap hits, against `accepted_answer_integers`. The scorer
uses the last complete nonempty balanced `\\boxed{...}` and requires one
integer in 0–999. Preserve the two-accepted-answer handling represented in the
transferred row. Sampling count and decoding parameters are intentionally not
pinned here; the Hyak evaluation implementation will pin and record those.

Produce two correct-versus-incorrect tables:

1. The two held-in-trained checkpoints evaluated on held-in (exact training
   problem distribution) and held-out (unseen problems).
2. The two held-out-trained checkpoints evaluated on held-out (exact training
   problem distribution) and held-in (unseen problems).

For every cell report accuracy, valid-box rate, cap-hit rate, and response-length
diagnostics. Include the base on both 240-problem sets and report
incorrect-minus-correct plus trained-minus-base deltas. Training datasets and
optimizer states are not part of this handoff.
"""


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
    if tuple(contract["variants"]) != VARIANTS:
        raise ValueError("contract variant order drift")
    optimizer_updates = int(contract["training"]["optimizer_updates"])
    final_iteration = int(contract["training"]["final_iteration"])
    if final_iteration != optimizer_updates - 1:
        raise ValueError("training update/final-iteration contract drift")
    handoff = root / "handoff"
    status_path = root / "TRAINING_STATUS.json"
    inventory_path = handoff / "checkpoint_sources.json"
    comparison_path = handoff / "comparison_sources.json"
    eval_contract_path = handoff / "eval_contract.json"
    eval_readme_path = handoff / "EVAL_HANDOFF.md"
    metrics_path = handoff / "training_metrics.json"
    for output in (
        status_path,
        inventory_path,
        comparison_path,
        eval_contract_path,
        eval_readme_path,
        metrics_path,
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite final artifact: {output}")
    chain_path = root / "manifests/training_chain.json"
    chain = load_json(chain_path)
    jobs = chain.get("jobs_in_dependency_order")
    if not isinstance(jobs, list) or len(jobs) != 10:
        raise ValueError("training chain must record exactly ten serial jobs")
    if str(jobs[-1]["job_id"]) != str(args.job_id):
        raise ValueError("running finalizer does not match training-chain tail")
    jobs_by_variant = {
        str(job["variant"]): str(job["job_id"]) for job in jobs if str(job.get("name", "")).startswith("train_")
    }
    if tuple(jobs_by_variant) != VARIANTS:
        raise ValueError(f"training job variant order drift: {tuple(jobs_by_variant)}")
    required_evidence = [
        root / "data/schedule_audit.json",
        root / "data/selection_and_tokenization_stats.json",
        root / "data/source_artifacts.json",
        root / "data/tokenizer_control_inventory.json",
        chain_path,
    ]
    for variant in VARIANTS:
        required_evidence.extend(
            [
                root / f"outputs/canaries/{variant}/evidence/before_runtime_token_audit.json",
                root / f"outputs/canaries/{variant}/evidence/after_runtime_token_audit.json",
            ]
        )
    for path in required_evidence:
        if not path.is_file():
            raise FileNotFoundError(f"missing completed experiment evidence: {path}")
    tool = args.repo_root / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
    gate = args.repo_root / "examples/qwen2_5_7b_aime_generalization_incorrect_sft/hf_checkpoint_gate.py"
    identities: dict[str, str] = {}
    transfer_records = []
    remote_experiment = f"{REMOTE_FAMILY}/{args.contract_hash}"
    for variant in VARIANTS:
        checkpoint = checkpoint_path(root, variant, final_iteration)
        manifest = handoff / "checkpoints" / f"{variant}.json"
        value = verify_checkpoint(tool=tool, gate=gate, manifest=manifest, checkpoint=checkpoint)
        expected = {
            "contract_hash": args.contract_hash,
            "family": "aime_2009_2024_split",
            "final_iteration": final_iteration,
            "full_sft": True,
            "variant": variant,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ValueError(f"checkpoint metadata drift: {variant}/{key}")
        if "parent_checkpoint_manifest_sha256" in value or "fresh_optimizer" in value:
            raise ValueError(f"{variant} was not trained independently from base")
        identity = str(value["checkpoint_manifest_sha256"])
        identities[variant] = identity
        transfer_records.append(
            {
                "checkpoint_identity": identity,
                "id": variant,
                "kind": "new_trained",
                "model_key": "qwen2_5_7b",
                "remote_checkpoint_relative": (f"outputs/training/{variant}/weights/iter_{final_iteration:07d}"),
                "remote_experiment_relative": remote_experiment,
                "remote_manifest_relative": f"handoff/checkpoints/{variant}.json",
                "source_checkpoint": str(checkpoint),
                "source_manifest": str(manifest),
                "variant": variant,
            }
        )
    base = verify_checkpoint(
        tool=tool,
        gate=gate,
        manifest=args.base_manifest,
        checkpoint=args.base_checkpoint,
    )
    base_identity = str(base["checkpoint_manifest_sha256"])
    if base_identity != contract["base_model"]["checkpoint_identity"]:
        raise ValueError("base checkpoint identity drift")
    copied_base = handoff / "checkpoints/base_qwen2_5_7b.json"
    copied_base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.base_manifest, copied_base)
    base_target = {
        "checkpoint_identity": base_identity,
        "id": BASE_ID,
        "kind": "existing_base",
        "model_key": "qwen2_5_7b",
        "remote_checkpoint_relative": "models/Qwen2.5-7B",
        "remote_experiment_relative": "checkpoints/shared",
        "remote_manifest_relative": "manifests/base_qwen2_5_7b.json",
        "source_checkpoint": str(args.base_checkpoint.resolve()),
        "source_manifest": str(args.base_manifest.resolve()),
        "variant": "base",
    }
    eval_sources = []
    eval_problem_ids: dict[str, set[str]] = {}
    for split in ("held_in", "held_out"):
        path = root / f"data/eval/{split}_240.jsonl"
        rows = read_jsonl(path)
        problem_ids = {str(row["problem_id"]) for row in rows}
        if len(rows) != 240 or len(problem_ids) != 240:
            raise ValueError(f"{split} evaluation source must have exactly 240 unique problems")
        if {str(row["split"]) for row in rows} != {split}:
            raise ValueError(f"{split} evaluation label drift")
        eval_problem_ids[split] = problem_ids
        eval_sources.append(
            {
                "path": f"data/eval/{split}_240.jsonl",
                "rows": len(rows),
                "sha256": sha256_file(path),
                "split": split,
            }
        )
    if (
        eval_problem_ids["held_in"] & eval_problem_ids["held_out"]
        or len(eval_problem_ids["held_in"] | eval_problem_ids["held_out"]) != 480
    ):
        raise ValueError("evaluation problem sets are not disjoint and exhaustive")
    scorer = args.repo_root / "examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval/aime_scorer.py"
    atomic_json(
        eval_contract_path,
        {
            "aggregation": "to be pinned by the Hyak evaluation implementation",
            "artifact_schema_version": 1,
            "cap_hits_scored": True,
            "checkpoints": [BASE_ID, *VARIANTS],
            "evaluation_sources": eval_sources,
            "prompt": {
                "messages": [{"role": "user", "content": "{prompt verbatim}"}],
                "rendered_prompt_field_used_directly": True,
                "system": None,
                "uses_qwen2_5_default_chat_template": False,
            },
            "reporting": {
                "base_on_both_splits": True,
                "correct_vs_incorrect_columns": True,
                "held_in_trained_rows": ["held_in_eval_in_distribution", "held_out_eval_unseen"],
                "held_out_trained_rows": ["held_out_eval_in_distribution", "held_in_eval_unseen"],
            },
            "sampling": "deferred_to_Hyak_and_must_be_recorded",
            "scorer": {
                "path_in_repo": str(scorer.relative_to(args.repo_root)),
                "sha256": sha256_file(scorer),
                "version": "aime_last_boxed_integer_scorer_v1",
            },
            "split_counts": {"held_in": 240, "held_out": 240},
        },
    )
    atomic_text(eval_readme_path, render_eval_handoff())
    training_metrics = []
    for variant in VARIANTS:
        job_id = jobs_by_variant[variant]
        log_path = root / "slurm_logs" / f"q25a6k-train_{job_id}.log"
        history = parse_metrics(log_path, final_iteration)
        training_metrics.append(
            {
                "first": history[0],
                "history": history,
                "job_id": job_id,
                "last": history[-1],
                "loss_change": history[-1]["loss"] - history[0]["loss"],
                "source_log": str(log_path),
                "source_log_sha256": sha256_file(log_path),
                "variant": variant,
            }
        )
    atomic_json(
        metrics_path,
        {
            "artifact_schema_version": 1,
            "optimizer_updates_per_variant": optimizer_updates,
            "variants": training_metrics,
        },
    )
    lightweight = [
        "TRAINING_STATUS.json",
        "data/eval/held_in_240.jsonl",
        "data/eval/held_out_240.jsonl",
        "data/schedule_audit.json",
        "data/selection_and_tokenization_stats.json",
        "data/source_artifacts.json",
        "data/tokenizer_control_inventory.json",
        "handoff/EVAL_HANDOFF.md",
        "handoff/checkpoint_sources.json",
        "handoff/comparison_sources.json",
        "handoff/eval_contract.json",
        "handoff/training_metrics.json",
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
            "comparison_checkpoint_count": 5,
            "contract_hash": args.contract_hash,
            "targets": [base_target, *transfer_records],
        },
    )
    commit = subprocess.check_output(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip()
    atomic_json(
        status_path,
        {
            "artifact_schema_version": 1,
            "base_checkpoint_reused": 1,
            "checkpoint_identities": {BASE_ID: base_identity, **identities},
            "comparison_checkpoint_count": 5,
            "contract_hash": args.contract_hash,
            "final_iterations": {variant: final_iteration for variant in VARIANTS},
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "full_parameter_sft": True,
            "repo_commit": commit,
            "required_evidence_sha256": {str(path.relative_to(root)): sha256_file(path) for path in required_evidence},
            "trained_checkpoints": 4,
            "training_datasets_transferred": False,
            "transfer_checkpoint_count": 4,
        },
    )
    print(
        "AIME_SPLIT_6K_TRAINING_HANDOFF_READY " f"trained=4 compared=5 contract={args.contract_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
