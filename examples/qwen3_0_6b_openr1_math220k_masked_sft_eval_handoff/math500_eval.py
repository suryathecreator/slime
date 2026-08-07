#!/usr/bin/env python3
"""Prepare, validate, score, merge, and audit the 0.6B MATH-500 eval."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen3_8b_openr1_math220k_masked_sft_eval_handoff import (
    axolotl_math500_eval as native,
)
from examples.qwen3_8b_32b_full_repro.math500_scorer_v5 import (
    SCORER_VERSION,
    TRACE_SCHEMA_VERSION,
    interpret_answer,
    score_response,
    verify_symmetric,
)


HANDOFF = Path(__file__).resolve().parent
MANIFEST = HANDOFF / "available_eval_manifest.json"
POLICY = HANDOFF / "eval_policy.json"
SCORER = (
    HANDOFF.parent
    / "qwen3_8b_32b_full_repro"
    / "math500_scorer_v5.py"
)
EXPECTED_VARIANTS = {
    "base_0p6b",
    "correct_only",
    "unmasked",
    "random_mask_25",
    "random_mask_50",
    "random_mask_70",
    "random_mask_80",
    "random_mask_90",
    "inverse_tau_0p20",
    "inverse_tau_0p05",
    "margin_mask",
    "prob_ratio_mask",
}
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
TOKENIZER_FILES = {
    "chat_template.jinja": (
        "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"
    ),
    "tokenizer.json": (
        "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
    ),
    "tokenizer_config.json": (
        "cbea7bca9904d56693d2226b00fd564e3be053609d1e50f8f1885510f0f6e790"
    ),
}


def configure_native() -> None:
    native.HANDOFF = HANDOFF
    native.DEFAULT_MANIFEST = MANIFEST
    native.POLICY = POLICY
    native.EXPECTED_VARIANTS = EXPECTED_VARIANTS
    native.EXPECTED_CHECKPOINT_COUNT = len(EXPECTED_VARIANTS)
    native.APPROVED_BASE_REVISION = BASE_REVISION
    native.EXPECTED_BASE_TOKENIZER_FILES = TOKENIZER_FILES
    native.prepare_tokenizer_overlay = prepare_tokenizer_overlay


def _inventory(repo_root: Path, manifest: str | Path = MANIFEST) -> dict[str, Any]:
    return native.load_inventory(repo_root, manifest)


def _paths(repo_root: Path, item: dict[str, Any]) -> dict[str, Path]:
    return native.absolute_item_paths(repo_root, item)


def prepare_tokenizer_overlay(
    repo_root: Path, inventory: dict[str, Any]
) -> Path:
    """Build one tokenizer overlay and prove its relation to the base tokenizer."""
    from tokenizers import Tokenizer

    source_item = native.item_for_variant(inventory, "correct_only")
    source_paths = _paths(repo_root, source_item)
    source = source_paths["model"]
    source_manifest = native.load_json(source_paths["manifest"])
    source_hashes = {
        entry["name"]: entry["sha256"]
        for entry in source_manifest["files"]
    }
    if {name: source_hashes.get(name) for name in TOKENIZER_FILES} != TOKENIZER_FILES:
        raise RuntimeError("trained tokenizer files differ from the approved pin")
    for item in inventory["checkpoints"]:
        if item["variant"] == "base_0p6b":
            continue
        manifest = native.load_json(_paths(repo_root, item)["manifest"])
        hashes = {entry["name"]: entry["sha256"] for entry in manifest["files"]}
        if {name: hashes.get(name) for name in TOKENIZER_FILES} != TOKENIZER_FILES:
            raise RuntimeError(f"trained tokenizer differs: {item['variant']}")

    base_paths = _paths(
        repo_root, native.item_for_variant(inventory, "base_0p6b")
    )
    base_tokenizer_path = base_paths["model"] / "tokenizer.json"
    trained_tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    base_tokenizer = Tokenizer.from_file(str(base_tokenizer_path))
    trained_vocab = trained_tokenizer.get_vocab(with_added_tokens=True)
    base_vocab = base_tokenizer.get_vocab(with_added_tokens=True)
    changed_shared_ids = {
        token: (token_id, trained_vocab.get(token))
        for token, token_id in base_vocab.items()
        if trained_vocab.get(token) != token_id
    }
    if changed_shared_ids:
        raise RuntimeError(
            f"trained tokenizer changes base token IDs: {changed_shared_ids}"
        )
    trained_only = sorted(set(trained_vocab) - set(base_vocab))
    expected_trained_only = sorted(
        ["<think>", "</think>", "<tool_response>", "</tool_response>"]
    )
    if trained_only != expected_trained_only:
        raise RuntimeError(
            f"unexpected trained-only tokenizer entries: {trained_only}"
        )
    base_config = native.load_json(base_paths["model"] / "tokenizer_config.json")
    trained_template = (source / "chat_template.jinja").read_text(encoding="utf-8")
    if base_config.get("chat_template") != trained_template:
        raise RuntimeError("base and trained chat templates differ")

    destination = native.tokenizer_root(repo_root, inventory)
    trained_config = native.load_json(source / "tokenizer_config.json")
    extra_tokens = trained_config.get("extra_special_tokens")
    if not isinstance(extra_tokens, list):
        raise RuntimeError("trained tokenizer compatibility source changed")
    compatible_config = dict(trained_config)
    compatible_config["extra_special_tokens"] = {}
    for name in TOKENIZER_FILES:
        if name == "tokenizer_config.json":
            native.atomic_text(destination / name, native.canonical_json(compatible_config))
        else:
            native.atomic_bytes(destination / name, (source / name).read_bytes())
    provenance = {
        "artifact_schema_version": 1,
        "base_model": inventory["base_model"],
        "base_tokenizer": {
            "path": str(base_tokenizer_path),
            "sha256": native.sha256_file(base_tokenizer_path),
            "vocabulary_size": len(base_vocab),
        },
        "compatibility_transform": {
            "field": "extra_special_tokens",
            "from": extra_tokens,
            "to": {},
            "reason": "Transformers 4.57.6 requires a mapping; tokenizer.json retains the added tokens.",
        },
        "files": {
            name: {
                "destination_sha256": native.sha256_file(destination / name),
                "source_sha256": TOKENIZER_FILES[name],
            }
            for name in TOKENIZER_FILES
        },
        "shared_vocabulary_ids_equal": True,
        "trained_only_tokens": trained_only,
        "trained_tokenizer": {
            "path": str(source / "tokenizer.json"),
            "sha256": TOKENIZER_FILES["tokenizer.json"],
            "vocabulary_size": len(trained_vocab),
        },
    }
    native.atomic_text(
        destination / "TOKENIZER_PROVENANCE.json",
        native.canonical_json(provenance),
    )
    return destination


def verify_checkpoints(args: argparse.Namespace) -> None:
    repo_root = native.repo_root_from_args(args)
    inventory = _inventory(repo_root, args.manifest)
    identities: dict[str, str] = {}
    for item in inventory["checkpoints"]:
        paths = _paths(repo_root, item)
        identity = (
            native.checkpoint_manifests.verify(paths["manifest"], paths["model"])
            if args.full
            else native.quick_checkpoint_check(paths["model"], paths["manifest"])
        )
        manifest = native.load_json(paths["manifest"])
        if manifest.get("variant") != item["variant"]:
            raise RuntimeError(f"manifest variant mismatch: {item['variant']}")
        if item["variant"] == "base_0p6b":
            if manifest.get("family") != "shared" or manifest.get("final_iteration") is not None:
                raise RuntimeError("base manifest identity changed")
        elif (
            manifest.get("family") != "2k"
            or int(manifest.get("final_iteration", -1)) != 124
            or manifest.get("full_sft") is not True
        ):
            raise RuntimeError(f"not an approved final Full-SFT checkpoint: {item['variant']}")
        identities[item["variant"]] = identity
        print(
            f"CHECKPOINT_VERIFIED variant={item['variant']} full={int(args.full)} sha256={identity}",
            flush=True,
        )
    if args.status:
        native.atomic_text(
            Path(args.status),
            native.canonical_json(
                {
                    "artifact_schema_version": 1,
                    "checked_at": native.now_iso(),
                    "full_hash_verification": bool(args.full),
                    "identities": identities,
                }
            ),
        )


def validate_gold(args: argparse.Namespace) -> None:
    repo_root = native.repo_root_from_args(args)
    inventory = _inventory(repo_root, args.manifest)
    benchmark = native.benchmark_paths(repo_root, inventory)[0]
    rows = native.read_jsonl(benchmark)
    failures: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    for row in rows:
        gold = str(row["correct_answer"])
        interpretations = interpret_answer(gold)
        kinds.update(item.kind for item in interpretations)
        self_check = verify_symmetric((gold,), gold)
        if not interpretations or not self_check["equivalent"]:
            failures.append(
                {
                    "problem_id": row["problem_id"],
                    "gold": gold,
                    "interpretations": [item.kind for item in interpretations],
                }
            )
    if len(rows) != 500 or failures:
        raise RuntimeError(
            f"gold coverage failed rows={len(rows)} failures={failures}"
        )
    output = native.control_root(repo_root, inventory) / "GOLD_COVERAGE.json"
    native.atomic_text(
        output,
        native.canonical_json(
            {
                "artifact_schema_version": 1,
                "gold_conditioned_routing": False,
                "interpretation_counts": dict(sorted(kinds.items())),
                "rows": len(rows),
                "scorer_version": SCORER_VERSION,
                "status": "complete",
            }
        ),
    )
    print(f"MATH500_V5_GOLD_COVERAGE_COMPLETE rows=500 output={output}", flush=True)


def _score_rows(
    full_rows: list[dict[str, Any]],
    native_predictions: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    predictions = {str(row["problem_id"]): row for row in native_predictions}
    raw = {str(row["problem_id"]): row for row in raw_rows}
    result: list[dict[str, Any]] = []
    for benchmark_row in full_rows:
        problem_id = str(benchmark_row["problem_id"])
        prediction = predictions[problem_id]
        generation = raw[problem_id]
        cap_hit = bool(prediction["cap_hit"])
        trace = score_response(
            str(generation["generation"]),
            str(benchmark_row["correct_answer"]),
            finish_reason=generation.get("finish_reason"),
            stop_token_id=None,
            cap_hit=cap_hit,
            problem_id=problem_id,
        )
        result.append(
            {
                "cap_hit": cap_hit,
                "correct_answer": str(benchmark_row["correct_answer"]),
                "finish_reason": generation.get("finish_reason"),
                "generated_token_count": int(prediction["generated_token_count"]),
                "is_correct": bool(trace["is_correct"]),
                "is_scorable": bool(trace["is_scorable"]),
                "problem_id": problem_id,
                "score_trace": trace,
                "scorer_version": SCORER_VERSION,
                "variant": variant,
            }
        )
    return result


def _summary(
    item: dict[str, Any],
    checkpoint: dict[str, Any],
    inventory: dict[str, Any],
    policy_sha: str,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(row["is_correct"] for row in predictions)
    cap_hits = sum(row["cap_hit"] for row in predictions)
    cap_correct = sum(row["is_correct"] for row in predictions if row["cap_hit"])
    valid_box = sum(
        row["score_trace"]["selected_box_group_id"] is not None
        for row in predictions
    )
    malformed = sum(bool(row["score_trace"]["malformed_boxes"]) for row in predictions)
    no_box = sum(
        row["score_trace"]["selected_box_group_id"] is None
        for row in predictions
    )
    parse_failures = sum(
        row["score_trace"]["official_decision_reason"] == "parse_failure"
        for row in predictions
    )
    token_total = sum(row["generated_token_count"] for row in predictions)
    return {
        "artifact_schema_version": 1,
        "accuracy": correct / total,
        "average_generated_length": token_total / total,
        "benchmark": "HuggingFaceH4/MATH-500",
        "cap_hit_accuracy": cap_correct / cap_hits if cap_hits else 0.0,
        "cap_hit_correct": cap_correct,
        "cap_hit_rate": cap_hits / total,
        "cap_hits": cap_hits,
        "checkpoint_manifest_sha256": checkpoint["checkpoint_manifest_sha256"],
        "correct": correct,
        "dataset_revision": inventory["dataset"]["revision"],
        "malformed_box_count": malformed,
        "no_valid_box_count": no_box,
        "parse_failure_count": parse_failures,
        "per_problem": [
            {
                "cap_hit": row["cap_hit"],
                "generated_token_count": row["generated_token_count"],
                "is_correct": row["is_correct"],
                "is_scorable": row["is_scorable"],
                "official_decision_reason": row["score_trace"]["official_decision_reason"],
                "problem_id": row["problem_id"],
                "selected_components_raw": row["score_trace"]["official_extracted_components_raw"],
            }
            for row in predictions
        ],
        "policy_sha256": policy_sha,
        "scorer_sha256": native.sha256_file(SCORER),
        "scorer_version": SCORER_VERSION,
        "stage": item["id"],
        "total": total,
        "valid_box_count": valid_box,
        "valid_box_rate": valid_box / total,
        "variant": item["variant"],
    }


def _compact_files(
    summary: dict[str, Any],
    provenance: dict[str, Any],
    checkpoint: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, str]:
    markdown = (
        f"# {summary['stage']} MATH-500 ({SCORER_VERSION})\n\n"
        f"- Accuracy: {summary['correct']}/{summary['total']} ({summary['accuracy']:.2%})\n"
        f"- Valid boxed answers: {summary['valid_box_count']}/{summary['total']} ({summary['valid_box_rate']:.2%})\n"
        f"- Cap hits: {summary['cap_hits']}/{summary['total']} ({summary['cap_hit_rate']:.2%})\n"
        f"- Correct cap hits: {summary['cap_hit_correct']}/{summary['cap_hits']} ({summary['cap_hit_accuracy']:.2%})\n"
        f"- No valid box: {summary['no_valid_box_count']}\n"
        f"- Parse failures: {summary['parse_failure_count']}\n"
        f"- Checkpoint: `{summary['checkpoint_manifest_sha256']}`\n"
    )
    return {
        "CHECKPOINT.json": native.canonical_json(checkpoint),
        "EVAL_POLICY.json": native.canonical_json(policy),
        "PROVENANCE.json": native.canonical_json(provenance),
        "RESULTS.json": native.canonical_json(summary),
        "RESULTS.md": markdown,
    }


def merge(args: argparse.Namespace) -> None:
    repo_root = native.repo_root_from_args(args)
    inventory = _inventory(repo_root, args.manifest)
    item = native.item_for_variant(inventory, args.variant)
    paths = _paths(repo_root, item)
    full_eval, shard_files, metadata_path = native.benchmark_paths(repo_root, inventory)
    metadata = native.load_json(metadata_path)
    if metadata["full"]["sha256"] != inventory["dataset"]["axolotl_canonical_jsonl_sha256"]:
        raise RuntimeError("prepared benchmark metadata hash mismatch")
    full_rows = native.read_jsonl(full_eval)
    native_predictions: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    shard_metrics: list[dict[str, Any]] = []
    for index, shard_file in enumerate(shard_files):
        shard_output = paths["output"] / "shards" / f"shard_{index:02d}"
        predictions, raw, metrics = native.validate_native_shard(
            eval_file=shard_file,
            output_dir=shard_output,
            variant=item["variant"],
        )
        native_predictions.extend(predictions)
        raw_rows.extend(raw)
        shard_metrics.append(metrics)
    expected_ids = [str(row["problem_id"]) for row in full_rows]
    if [str(row["problem_id"]) for row in native_predictions] != expected_ids:
        raise RuntimeError("native prediction order differs from MATH-500")
    if [str(row["problem_id"]) for row in raw_rows] != expected_ids:
        raise RuntimeError("raw generation order differs from MATH-500")
    official = _score_rows(full_rows, native_predictions, raw_rows, item["variant"])
    checkpoint = native.load_json(paths["manifest"])
    native.checkpoint_manifests.validate_manifest(checkpoint)
    policy = native.load_json(POLICY)
    policy_sha = native.sha256_bytes(native.canonical_json(policy).encode())
    summary = _summary(item, checkpoint, inventory, policy_sha, official)
    if summary["total"] != 500:
        raise RuntimeError("official scorer did not produce exactly 500 rows")
    merged = paths["output"] / "merged"
    native.atomic_text(merged / "predictions.jsonl", native.jsonl_text(official))
    native.atomic_text(merged / "native_predictions.jsonl", native.jsonl_text(native_predictions))
    native.atomic_text(merged / "raw_generations.jsonl", native.jsonl_text(raw_rows))
    native.atomic_text(merged / "summary.json", native.canonical_json(summary))
    native.atomic_text(merged / "summary.sha256", native.sha256_file(merged / "summary.json") + "\n")
    provenance = {
        "artifact_schema_version": 1,
        "axolotl": inventory["axolotl"],
        "checkpoint_manifest": str(paths["manifest"]),
        "checkpoint_manifest_sha256": checkpoint["checkpoint_manifest_sha256"],
        "dataset_jsonl": str(full_eval),
        "dataset_jsonl_sha256": native.sha256_file(full_eval),
        "native_shard_metrics": shard_metrics,
        "policy_sha256": policy_sha,
        "raw_results": str(merged),
        "scorer_path": str(SCORER),
        "scorer_sha256": native.sha256_file(SCORER),
        "scorer_version": SCORER_VERSION,
        "stage": item["id"],
        "tokenizer_provenance_sha256": native.sha256_file(
            native.tokenizer_root(repo_root, inventory) / "TOKENIZER_PROVENANCE.json"
        ),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
    }
    native.atomic_text(merged / "provenance.json", native.canonical_json(provenance))
    native.install_compact_result(
        paths["result"],
        _compact_files(summary, provenance, checkpoint, policy),
    )
    print(
        f"MATH500_V5_MERGE_COMPLETE variant={item['variant']} correct={summary['correct']}/500 "
        f"cap_hits={summary['cap_hits']} cap_correct={summary['cap_hit_correct']} result={paths['result']}",
        flush=True,
    )


def audit(args: argparse.Namespace) -> None:
    repo_root = native.repo_root_from_args(args)
    inventory = _inventory(repo_root, args.manifest)
    rows: list[dict[str, Any]] = []
    for item in inventory["checkpoints"]:
        paths = _paths(repo_root, item)
        result = native.load_json(paths["result"] / "RESULTS.json")
        merged = native.load_json(paths["output"] / "merged" / "summary.json")
        if result != merged or result.get("total") != 500:
            raise RuntimeError(f"incomplete or divergent result: {item['variant']}")
        if result.get("scorer_version") != SCORER_VERSION:
            raise RuntimeError(f"wrong scorer version: {item['variant']}")
        rows.append(
            {
                "accuracy": result["accuracy"],
                "cap_hit_accuracy": result["cap_hit_accuracy"],
                "cap_hit_correct": result["cap_hit_correct"],
                "cap_hit_rate": result["cap_hit_rate"],
                "cap_hits": result["cap_hits"],
                "correct": result["correct"],
                "parse_failure_count": result["parse_failure_count"],
                "stage": result["stage"],
                "valid_box_rate": result["valid_box_rate"],
                "variant": result["variant"],
            }
        )
    output_root = native.control_root(repo_root, inventory)
    audit_value = {
        "artifact_schema_version": 1,
        "audited_at": native.now_iso(),
        "count": len(rows),
        "dataset": inventory["dataset"],
        "results": rows,
        "scorer_version": SCORER_VERSION,
        "status": "complete",
    }
    native.atomic_text(output_root / "FINAL_AUDIT.json", native.canonical_json(audit_value))
    report_root = repo_root / "examples/qwen3_0_6b_openr1_math220k_masked_sft_2k/results/math500_v5"
    native.atomic_text(report_root / "RESULTS.json", native.canonical_json(audit_value))
    lines = [
        "# Qwen3-0.6B MATH-500 V5 results",
        "",
        "| Checkpoint | Correct | Accuracy | Valid box | Cap hits | "
        "Correct cap hits | Cap-hit accuracy | Parse failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['correct']}/500 | {row['accuracy']:.2%} | "
            f"{row['valid_box_rate']:.2%} | {row['cap_hits']} ({row['cap_hit_rate']:.2%}) | "
            f"{row['cap_hit_correct']} | {row['cap_hit_accuracy']:.2%} | {row['parse_failure_count']} |"
        )
    native.atomic_text(report_root / "RESULTS.md", "\n".join(lines) + "\n")
    print(f"MATH500_V5_ALL_RESULTS_COMPLETE count={len(rows)} audit={output_root / 'FINAL_AUDIT.json'}", flush=True)


def mark_canary(args: argparse.Namespace) -> None:
    repo_root = native.repo_root_from_args(args)
    inventory = _inventory(repo_root, args.manifest)
    root = native.control_root(repo_root, inventory)
    rows = native.read_jsonl(root / "eval_benchmarks/smoke.jsonl")
    raw = native.read_jsonl(root / "canary/raw_generations.jsonl")
    predictions = native.read_jsonl(root / "canary/predictions.jsonl")
    if len(rows) != len(raw) or len(rows) != len(predictions) or len(rows) != 1:
        raise RuntimeError("canary must contain exactly one complete generation")
    trace = score_response(
        str(raw[0]["generation"]),
        str(rows[0]["correct_answer"]),
        finish_reason=raw[0].get("finish_reason"),
        stop_token_id=None,
        cap_hit=bool(predictions[0]["cap_hit"]),
        problem_id=rows[0]["problem_id"],
    )
    if trace["scorer_version"] != SCORER_VERSION:
        raise RuntimeError("canary scorer version mismatch")
    ready = {
        "artifact_schema_version": 1,
        "job_id": args.job_id,
        "ready_at": native.now_iso(),
        "scorer_version": SCORER_VERSION,
        "status": "ready",
        "trace": trace,
    }
    native.atomic_text(root / "CANARY_READY.json", native.canonical_json(ready))
    print(f"MATH500_V5_CANARY_READY output={root / 'CANARY_READY.json'}", flush=True)


def check_canary(args: argparse.Namespace) -> None:
    repo_root = native.repo_root_from_args(args)
    inventory = _inventory(repo_root, args.manifest)
    root = native.control_root(repo_root, inventory)
    native.check_preflight(args)
    ready = native.load_json(root / "CANARY_READY.json")
    if ready.get("status") != "ready" or ready.get("scorer_version") != SCORER_VERSION:
        raise RuntimeError("generation/scoring canary is not ready")
    print(f"MATH500_V5_CANARY_VERIFIED status={root / 'CANARY_READY.json'}", flush=True)


def record_submission(args: argparse.Namespace) -> None:
    repo_root = native.repo_root_from_args(args)
    inventory = _inventory(repo_root, args.manifest)
    output = native.control_root(repo_root, inventory) / "submission.json"
    if output.exists():
        raise FileExistsError(f"refusing duplicate submission: {output}")

    def assignments(values: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for value in values:
            name, separator, job_id = value.partition("=")
            if separator != "=" or not name or not job_id or name in result:
                raise ValueError(f"invalid assignment: {value}")
            result[name] = job_id
        return result

    arrays = assignments(args.array_job)
    merges = assignments(args.merge_job)
    if set(arrays) != EXPECTED_VARIANTS or set(merges) != EXPECTED_VARIANTS:
        raise RuntimeError("submission journal does not cover all checkpoints")
    native.atomic_text(
        output,
        native.canonical_json(
            {
                "artifact_schema_version": 1,
                "array_jobs": arrays,
                "audit_job": args.audit_job,
                "canary_job": args.canary_job,
                "checkpoint_count": len(EXPECTED_VARIANTS),
                "eval_task_count": len(EXPECTED_VARIANTS) * 4,
                "merge_jobs": merges,
                "preflight_job": args.preflight_job,
                "slurm": {
                    "account": "raivn-ckpt",
                    "array": "0-3",
                    "partition": "ckpt-all",
                    "qos": "ckpt",
                    "throttle": None,
                    "walltime": "02:00:00",
                },
                "submitted_at": native.now_iso(),
            }
        ),
    )
    print(f"MATH500_V5_SUBMISSION_RECORDED journal={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST))
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source")
    verify_parser = subparsers.add_parser("verify-checkpoints")
    verify_parser.add_argument("--full", action="store_true")
    verify_parser.add_argument("--status")
    subparsers.add_parser("validate-gold")
    repair_parser = subparsers.add_parser("repair-shard")
    repair_parser.add_argument("--eval-file", required=True)
    repair_parser.add_argument("--output-dir", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--variant", required=True)
    subparsers.add_parser("audit")
    preflight_parser = subparsers.add_parser("mark-preflight")
    preflight_parser.add_argument("--job-id", required=True)
    subparsers.add_parser("check-preflight")
    canary_parser = subparsers.add_parser("mark-canary")
    canary_parser.add_argument("--job-id", required=True)
    subparsers.add_parser("check-canary")
    record_parser = subparsers.add_parser("record-submission")
    record_parser.add_argument("--preflight-job", required=True)
    record_parser.add_argument("--canary-job", required=True)
    record_parser.add_argument("--array-job", action="append", default=[])
    record_parser.add_argument("--merge-job", action="append", default=[])
    record_parser.add_argument("--audit-job", required=True)
    path_parser = subparsers.add_parser("path")
    path_parser.add_argument(
        "--field",
        choices=[
            "model",
            "manifest",
            "output",
            "result",
            "eval_root",
            "control_root",
            "tokenizer",
            "benchmark",
            "shard",
        ],
        required=True,
    )
    path_parser.add_argument("--variant")
    path_parser.add_argument("--shard-index", type=int, choices=range(4))
    return parser


def main() -> None:
    configure_native()
    args = build_parser().parse_args()
    handlers = {
        "prepare": native.prepare,
        "verify-checkpoints": verify_checkpoints,
        "validate-gold": validate_gold,
        "repair-shard": native.repair_shard,
        "merge": merge,
        "audit": audit,
        "mark-preflight": native.mark_preflight,
        "check-preflight": native.check_preflight,
        "mark-canary": mark_canary,
        "check-canary": check_canary,
        "record-submission": record_submission,
        "path": native.print_path,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
