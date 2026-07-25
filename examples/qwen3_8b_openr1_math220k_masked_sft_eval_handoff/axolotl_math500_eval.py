#!/usr/bin/env python3
"""Prepare, validate, merge, and audit the available Axolotl MATH-500 evals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from . import checkpoint_manifest as checkpoint_manifests
except ImportError:
    import checkpoint_manifest as checkpoint_manifests


HANDOFF = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HANDOFF / "available_eval_manifest.json"
POLICY = HANDOFF / "axolotl_eval_policy.json"
EXPECTED_TOTAL = 500
EXPECTED_SHARDS = 4
EXPECTED_VARIANTS = {
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
DEFAULT_AXOLOTL_MATH500 = Path(
    "../Axolotl-Masked-SFT/runs/"
    "2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm/"
    "eval_benchmarks/math500.jsonl"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temp = Path(handle.name)
        temp.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def repo_root_from_args(args: argparse.Namespace) -> Path:
    root = Path(args.repo_root).resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"not a Git worktree: {root}")
    return root


def load_inventory(
    repo_root: Path, manifest_path: str | Path = DEFAULT_MANIFEST
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        if Path(manifest_path) == DEFAULT_MANIFEST:
            path = DEFAULT_MANIFEST
        else:
            raise FileNotFoundError(path)
    value = load_json(path)
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 11:
        raise RuntimeError("available eval manifest must contain 11 checkpoints")
    variants = [str(item["variant"]) for item in checkpoints]
    if len(set(variants)) != len(variants) or set(variants) != EXPECTED_VARIANTS:
        raise RuntimeError(
            "available checkpoint variants differ from the approved eleven"
        )
    if value["base_model"]["revision"] != (
        "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    ):
        raise RuntimeError("base model revision is not the approved pin")
    if value["dataset"]["revision"] != (
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    ):
        raise RuntimeError("MATH-500 revision is not the approved pin")
    if value["axolotl"] != {
        "commit": "6b8f0e3314e3d162260cdc35d84741c3da163f30",
        "evaluator": "scripts/openr1_axolotl/evaluate_math_vllm.py",
        "repository": "../Axolotl-Masked-SFT",
    }:
        raise RuntimeError("Axolotl evaluator provenance is not the approved pin")
    if value["runtime"] != {
        "math_verify": "0.9.0",
        "torch": "2.8.0",
        "transformers": "4.57.6",
        "vllm": "0.10.2",
    }:
        raise RuntimeError("Axolotl runtime is not the approved pin")
    if value["sharding"] != {
        "chunk_size": 32,
        "count": EXPECTED_SHARDS,
        "layout": "contiguous",
        "problems_per_shard": 125,
    }:
        raise RuntimeError("approved run requires exactly four shards")
    return value


def item_for_variant(
    inventory: dict[str, Any], variant: str
) -> dict[str, Any]:
    matches = [
        item for item in inventory["checkpoints"] if item["variant"] == variant
    ]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate variant: {variant}")
    return matches[0]


def absolute_item_paths(
    repo_root: Path, item: dict[str, Any]
) -> dict[str, Path]:
    return {
        key: (repo_root / item[key]).resolve()
        for key in ("manifest", "model", "output", "result")
    }


def eval_root(repo_root: Path, inventory: dict[str, Any]) -> Path:
    parents = {
        absolute_item_paths(repo_root, item)["output"].parent
        for item in inventory["checkpoints"]
    }
    if len(parents) != 1:
        raise RuntimeError(f"checkpoint outputs do not share one eval root: {parents}")
    return next(iter(parents))


def control_root(repo_root: Path, inventory: dict[str, Any]) -> Path:
    return eval_root(repo_root, inventory) / "_control"


def benchmark_paths(
    repo_root: Path, inventory: dict[str, Any]
) -> tuple[Path, list[Path], Path]:
    root = control_root(repo_root, inventory) / "eval_benchmarks"
    full = root / "math500.jsonl"
    shards = [
        root / f"shard_{index:02d}_{index * 125:03d}_{(index + 1) * 125:03d}.jsonl"
        for index in range(EXPECTED_SHARDS)
    ]
    return full, shards, root / "metadata.json"


def normalize_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"MATH-500 source must have 500 rows, found {len(rows)}"
        )
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        problem = row.get("problem")
        answer = row.get("correct_answer", row.get("answer"))
        problem_id = (
            row.get("problem_id")
            or row.get("unique_id")
            or row.get("id")
            or f"math500_{index:03d}"
        )
        if not isinstance(problem, str) or not problem.strip():
            raise RuntimeError(f"MATH-500 row {index} has no problem")
        if answer is None:
            raise RuntimeError(f"MATH-500 row {index} has no answer")
        normalized.append(
            {
                "correct_answer": str(answer),
                "level": row.get("level"),
                "problem": problem,
                "problem_id": str(problem_id),
                "subject": row.get("subject"),
            }
        )
    identifiers = [row["problem_id"] for row in normalized]
    if len(set(identifiers)) != EXPECTED_TOTAL:
        raise RuntimeError("MATH-500 source contains duplicate problem IDs")
    return normalized


def prepare(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_inventory(repo_root, args.manifest)
    source = (
        Path(args.source).resolve()
        if args.source
        else (repo_root / DEFAULT_AXOLOTL_MATH500).resolve()
    )
    source_rows = read_jsonl(source)
    rows = normalize_source_rows(source_rows)
    full, shards, metadata_path = benchmark_paths(repo_root, inventory)
    full_text = jsonl_text(rows)
    expected_hash = inventory["dataset"]["axolotl_canonical_jsonl_sha256"]
    actual_hash = sha256_bytes(full_text.encode())
    if actual_hash != expected_hash:
        raise RuntimeError(
            "canonical Axolotl MATH-500 JSONL hash mismatch: "
            f"expected={expected_hash} actual={actual_hash} source={source}"
        )
    atomic_text(full, full_text)
    shard_records: list[dict[str, Any]] = []
    for index, target in enumerate(shards):
        start, end = index * 125, (index + 1) * 125
        shard_text = jsonl_text(rows[start:end])
        atomic_text(target, shard_text)
        shard_records.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "count": end - start,
                "path": str(target),
                "sha256": sha256_bytes(shard_text.encode()),
                "first_problem_id": rows[start]["problem_id"],
                "last_problem_id": rows[end - 1]["problem_id"],
            }
        )
    atomic_text(
        full.parent / "smoke.jsonl",
        jsonl_text(rows[:1]),
    )
    metadata = {
        "artifact_schema_version": 1,
        "created_at": now_iso(),
        "dataset": inventory["dataset"],
        "full": {
            "path": str(full),
            "count": EXPECTED_TOTAL,
            "sha256": actual_hash,
        },
        "shards": shard_records,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
    }
    atomic_text(metadata_path, canonical_json(metadata))
    print(
        f"AXOLOTL_MATH500_SHARDS_PREPARED rows=500 "
        f"sha256={actual_hash} root={full.parent}",
        flush=True,
    )


def quick_checkpoint_check(
    checkpoint: Path, manifest_path: Path
) -> str:
    value = load_json(manifest_path)
    checkpoint_manifests.validate_manifest(value)
    expected = {item["name"]: int(item["bytes"]) for item in value["files"]}
    actual = {
        path.relative_to(checkpoint).as_posix(): path.stat().st_size
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            name
            for name in set(actual) & set(expected)
            if actual[name] != expected[name]
        )
        raise RuntimeError(
            f"checkpoint size inventory mismatch at {checkpoint}: "
            f"missing={missing} extra={extra} changed={changed}"
        )
    return str(value["checkpoint_manifest_sha256"])


def verify_checkpoints(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_inventory(repo_root, args.manifest)
    identities: dict[str, str] = {}
    for item in inventory["checkpoints"]:
        paths = absolute_item_paths(repo_root, item)
        if args.full:
            identity = checkpoint_manifests.verify(
                paths["manifest"], paths["model"]
            )
        else:
            identity = quick_checkpoint_check(
                paths["model"], paths["manifest"]
            )
        manifest = load_json(paths["manifest"])
        if manifest.get("family") != "2k":
            raise RuntimeError(f"{item['variant']} is not a 2K checkpoint")
        if manifest.get("variant") != item["variant"]:
            raise RuntimeError(f"{item['variant']} manifest variant mismatch")
        if int(manifest.get("final_iteration", -1)) != 9:
            raise RuntimeError(f"{item['variant']} is not final iteration 9")
        identities[item["variant"]] = identity
        print(
            f"CHECKPOINT_VERIFIED variant={item['variant']} "
            f"full={int(args.full)} sha256={identity}",
            flush=True,
        )
    if args.status:
        status = {
            "artifact_schema_version": 1,
            "checked_at": now_iso(),
            "full_hash_verification": bool(args.full),
            "identities": identities,
        }
        atomic_text(Path(args.status), canonical_json(status))


def dedupe_latest(
    rows: list[dict[str, Any]], expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = str(row.get("problem_id"))
        if problem_id in expected_ids:
            result[problem_id] = row
    return result


def repair_shard(args: argparse.Namespace) -> None:
    eval_file = Path(args.eval_file)
    output_dir = Path(args.output_dir)
    expected_rows = read_jsonl(eval_file)
    expected_ids = [str(row["problem_id"]) for row in expected_rows]
    if len(expected_ids) != 125 or len(set(expected_ids)) != 125:
        raise RuntimeError("eval shard must contain 125 unique problem IDs")
    expected_set = set(expected_ids)
    predictions_path = output_dir / "predictions.jsonl"
    raw_path = output_dir / "raw_generations.jsonl"
    metrics_path = output_dir / "metrics.json"
    predictions = dedupe_latest(read_jsonl(predictions_path), expected_set)
    raw = dedupe_latest(read_jsonl(raw_path), expected_set)
    common = set(predictions) & set(raw)
    repaired_predictions = [predictions[item] for item in expected_ids if item in common]
    repaired_raw = [raw[item] for item in expected_ids if item in common]
    if metrics_path.exists():
        if common != expected_set:
            raise RuntimeError(
                f"completed shard has incomplete JSONL files: {output_dir}"
            )
        metrics = load_json(metrics_path)
        if int(metrics.get("num_eval_problems", -1)) != 125:
            raise RuntimeError(f"completed shard metrics mismatch: {metrics_path}")
    if predictions_path.exists() or repaired_predictions:
        atomic_text(predictions_path, jsonl_text(repaired_predictions))
    if raw_path.exists() or repaired_raw:
        atomic_text(raw_path, jsonl_text(repaired_raw))
    print(
        f"AXOLOTL_SHARD_RESUME_REPAIRED output={output_dir} "
        f"completed={len(common)} expected=125",
        flush=True,
    )


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(
            f"{label} mismatch: expected={expected} actual={actual}"
        )


def validate_native_shard(
    *,
    eval_file: Path,
    output_dir: Path,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected_rows = read_jsonl(eval_file)
    expected_ids = [str(row["problem_id"]) for row in expected_rows]
    expected_set = set(expected_ids)
    predictions = read_jsonl(output_dir / "predictions.jsonl")
    raw = read_jsonl(output_dir / "raw_generations.jsonl")
    if len(predictions) != 125 or len(raw) != 125:
        raise RuntimeError(
            f"shard must contain 125 predictions and raw generations: {output_dir}"
        )
    pred_by_id = dedupe_latest(predictions, expected_set)
    raw_by_id = dedupe_latest(raw, expected_set)
    if set(pred_by_id) != expected_set or set(raw_by_id) != expected_set:
        raise RuntimeError(f"shard problem IDs differ from eval file: {output_dir}")
    ordered_predictions = [pred_by_id[item] for item in expected_ids]
    ordered_raw = [raw_by_id[item] for item in expected_ids]
    if any(row.get("variant") != variant for row in ordered_predictions):
        raise RuntimeError(f"prediction variant mismatch: {output_dir}")
    if any(row.get("variant") != variant for row in ordered_raw):
        raise RuntimeError(f"raw-generation variant mismatch: {output_dir}")
    expected_answers = {
        str(row["problem_id"]): str(row["correct_answer"])
        for row in expected_rows
    }
    if any(
        str(row.get("correct_answer")) != expected_answers[str(row["problem_id"])]
        for row in ordered_predictions
    ):
        raise RuntimeError(f"prediction gold-answer mismatch: {output_dir}")
    metrics = load_json(output_dir / "metrics.json")
    if int(metrics.get("num_eval_problems", -1)) != 125:
        raise RuntimeError(f"native metrics row count mismatch: {output_dir}")
    if metrics.get("benchmark") != "math500":
        raise RuntimeError(f"native metrics benchmark mismatch: {output_dir}")
    if metrics.get("evaluator") != "vllm":
        raise RuntimeError(f"native metrics evaluator mismatch: {output_dir}")
    if metrics.get("variant") != variant:
        raise RuntimeError(f"native metrics variant mismatch: {output_dir}")
    if metrics.get("load_mode") != "full_sft":
        raise RuntimeError(f"native metrics load mode mismatch: {output_dir}")
    if Path(str(metrics.get("eval_file"))).resolve() != eval_file.resolve():
        raise RuntimeError(f"native metrics eval-file mismatch: {output_dir}")
    correct = sum(bool(row["is_correct"]) for row in ordered_predictions)
    tokens = sum(int(row["generated_token_count"]) for row in ordered_predictions)
    caps = sum(bool(row["cap_hit"]) for row in ordered_predictions)
    parse_failures = sum(
        row.get("predicted_answer") is None for row in ordered_predictions
    )
    require_close(float(metrics["accuracy"]), correct / 125, "accuracy")
    require_close(
        float(metrics["average_generated_length"]),
        tokens / 125,
        "average generated length",
    )
    require_close(float(metrics["cap_hit_rate"]), caps / 125, "cap-hit rate")
    require_close(
        float(metrics["parse_failure_rate"]),
        parse_failures / 125,
        "parse-failure rate",
    )
    return ordered_predictions, ordered_raw, metrics


def expected_compact_files(
    *,
    summary: dict[str, Any],
    provenance: dict[str, Any],
    checkpoint: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, str]:
    markdown = (
        f"# {summary['stage']} Axolotl MATH-500\n\n"
        f"- Accuracy: {summary['correct']}/{summary['total']} "
        f"({summary['accuracy']:.2%})\n"
        f"- Cap hits: {summary['cap_hits']}\n"
        f"- Parse failures: {summary['parse_failures']}\n"
        f"- Evaluator: `{summary['evaluator']}`\n"
        f"- Checkpoint: `{summary['checkpoint_manifest_sha256']}`\n"
    )
    return {
        "RESULTS.json": canonical_json(summary),
        "RESULTS.md": markdown,
        "PROVENANCE.json": canonical_json(provenance),
        "CHECKPOINT.json": canonical_json(checkpoint),
        "EVAL_POLICY.json": canonical_json(policy),
    }


def install_compact_result(
    destination: Path, expected: dict[str, str]
) -> None:
    if destination.exists():
        extra = sorted(
            path.name
            for path in destination.iterdir()
            if path.is_file() and path.name not in expected
        )
        if extra:
            raise RuntimeError(
                f"result destination has unexpected files: {destination} {extra}"
            )
        for name, content in expected.items():
            path = destination / name
            if path.exists() and path.read_text(encoding="utf-8") != content:
                raise RuntimeError(
                    f"refusing to overwrite incompatible result: {path}"
                )
    for name, content in expected.items():
        path = destination / name
        if not path.exists():
            atomic_text(path, content)


def merge(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_inventory(repo_root, args.manifest)
    item = item_for_variant(inventory, args.variant)
    paths = absolute_item_paths(repo_root, item)
    full_eval, shard_files, metadata_path = benchmark_paths(repo_root, inventory)
    metadata = load_json(metadata_path)
    if metadata["full"]["sha256"] != inventory["dataset"][
        "axolotl_canonical_jsonl_sha256"
    ]:
        raise RuntimeError("prepared benchmark metadata hash mismatch")
    full_rows = read_jsonl(full_eval)
    all_predictions: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    native_metrics: list[dict[str, Any]] = []
    for index, eval_file in enumerate(shard_files):
        output_dir = paths["output"] / "shards" / f"shard_{index:02d}"
        predictions, raw, metrics = validate_native_shard(
            eval_file=eval_file,
            output_dir=output_dir,
            variant=item["variant"],
        )
        all_predictions.extend(predictions)
        all_raw.extend(raw)
        native_metrics.append(metrics)
    expected_ids = [str(row["problem_id"]) for row in full_rows]
    actual_ids = [str(row["problem_id"]) for row in all_predictions]
    raw_ids = [str(row["problem_id"]) for row in all_raw]
    if actual_ids != expected_ids or raw_ids != expected_ids:
        raise RuntimeError("merged shard order differs from canonical MATH-500")
    checkpoint = load_json(paths["manifest"])
    checkpoint_manifests.validate_manifest(checkpoint)
    policy = load_json(POLICY)
    policy_sha = sha256_bytes(canonical_json(policy).encode())
    correct = sum(bool(row["is_correct"]) for row in all_predictions)
    token_total = sum(
        int(row["generated_token_count"]) for row in all_predictions
    )
    cap_hits = sum(bool(row["cap_hit"]) for row in all_predictions)
    parse_failures = sum(
        row.get("predicted_answer") is None for row in all_predictions
    )
    finish_reasons = Counter(
        str(row.get("finish_reason")) for row in all_predictions
    )
    stage = str(item["id"])
    summary = {
        "artifact_schema_version": 1,
        "accuracy": correct / EXPECTED_TOTAL,
        "average_generated_length": token_total / EXPECTED_TOTAL,
        "benchmark": "HuggingFaceH4/MATH-500",
        "cap_hit_rate": cap_hits / EXPECTED_TOTAL,
        "cap_hits": cap_hits,
        "checkpoint_manifest_sha256": checkpoint[
            "checkpoint_manifest_sha256"
        ],
        "correct": correct,
        "dataset_revision": inventory["dataset"]["revision"],
        "evaluator": (
            "Axolotl-Masked-SFT/scripts/openr1_axolotl/"
            "evaluate_math_vllm.py"
        ),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "parse_failure_rate": parse_failures / EXPECTED_TOTAL,
        "parse_failures": parse_failures,
        "per_problem": [
            {
                "cap_hit": bool(row["cap_hit"]),
                "generated_token_count": int(row["generated_token_count"]),
                "is_correct": bool(row["is_correct"]),
                "predicted_answer": row.get("predicted_answer"),
                "problem_id": str(row["problem_id"]),
            }
            for row in all_predictions
        ],
        "policy_sha256": policy_sha,
        "stage": stage,
        "total": EXPECTED_TOTAL,
        "variant": item["variant"],
    }
    merged = paths["output"] / "merged"
    atomic_text(merged / "predictions.jsonl", jsonl_text(all_predictions))
    atomic_text(merged / "raw_generations.jsonl", jsonl_text(all_raw))
    atomic_text(merged / "summary.json", canonical_json(summary))
    atomic_text(
        merged / "summary.sha256",
        sha256_file(merged / "summary.json") + "\n",
    )
    provenance = {
        "artifact_schema_version": 1,
        "axolotl_commit": inventory["axolotl"]["commit"],
        "checkpoint_manifest": str(paths["manifest"]),
        "checkpoint_manifest_sha256": checkpoint[
            "checkpoint_manifest_sha256"
        ],
        "dataset_jsonl": str(full_eval),
        "dataset_jsonl_sha256": sha256_file(full_eval),
        "dataset_revision": inventory["dataset"]["revision"],
        "native_shard_metrics": native_metrics,
        "policy_sha256": policy_sha,
        "raw_results": str(merged),
        "stage": stage,
    }
    atomic_text(merged / "provenance.json", canonical_json(provenance))
    expected = expected_compact_files(
        summary=summary,
        provenance=provenance,
        checkpoint=checkpoint,
        policy=policy,
    )
    install_compact_result(paths["result"], expected)
    print(
        f"AXOLOTL_MATH500_MERGE_COMPLETE variant={item['variant']} "
        f"correct={correct}/500 accuracy={correct / 500:.4f} "
        f"result={paths['result']}",
        flush=True,
    )


def audit(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_inventory(repo_root, args.manifest)
    rows: list[dict[str, Any]] = []
    for item in inventory["checkpoints"]:
        paths = absolute_item_paths(repo_root, item)
        result = load_json(paths["result"] / "RESULTS.json")
        merged = load_json(paths["output"] / "merged" / "summary.json")
        if result != merged:
            raise RuntimeError(f"compact and merged results differ: {item['variant']}")
        if int(result.get("total", -1)) != EXPECTED_TOTAL:
            raise RuntimeError(f"incomplete result: {item['variant']}")
        if len(result.get("per_problem", [])) != EXPECTED_TOTAL:
            raise RuntimeError(f"incomplete per-problem result: {item['variant']}")
        rows.append(
            {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "result": str(paths["result"]),
                "stage": item["id"],
                "variant": item["variant"],
            }
        )
    output = control_root(repo_root, inventory) / "FINAL_AUDIT.json"
    value = {
        "artifact_schema_version": 1,
        "audited_at": now_iso(),
        "count": len(rows),
        "dataset": inventory["dataset"],
        "results": rows,
        "status": "complete",
    }
    atomic_text(output, canonical_json(value))
    print(f"AXOLOTL_ALL_RESULTS_COMPLETE count=11 audit={output}", flush=True)


def mark_preflight(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_inventory(repo_root, args.manifest)
    versions = {
        name: importlib.metadata.version(distribution)
        for name, distribution in (
            ("vllm", "vllm"),
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("math_verify", "math-verify"),
        )
    }
    if versions != inventory["runtime"]:
        raise RuntimeError(
            f"Axolotl evaluation runtime mismatch: {versions} "
            f"!= {inventory['runtime']}"
        )
    verification_path = (
        control_root(repo_root, inventory) / "checkpoint_verification.json"
    )
    verification = load_json(verification_path)
    if verification.get("full_hash_verification") is not True:
        raise RuntimeError("preflight requires full checkpoint hash verification")
    if set(verification.get("identities", {})) != EXPECTED_VARIANTS:
        raise RuntimeError("preflight checkpoint identities are incomplete")
    for item in inventory["checkpoints"]:
        manifest = load_json(absolute_item_paths(repo_root, item)["manifest"])
        if verification["identities"][item["variant"]] != manifest[
            "checkpoint_manifest_sha256"
        ]:
            raise RuntimeError(
                f"preflight identity mismatch: {item['variant']}"
            )
    value = {
        "artifact_schema_version": 1,
        "axolotl": inventory["axolotl"],
        "checkpoint_verification_sha256": sha256_file(verification_path),
        "job_id": args.job_id,
        "ready_at": now_iso(),
        "runtime": versions,
        "status": "ready",
    }
    output = control_root(repo_root, inventory) / "PREFLIGHT_READY.json"
    atomic_text(output, canonical_json(value))
    print(f"AXOLOTL_PREFLIGHT_READY status={output}", flush=True)


def check_preflight(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_inventory(repo_root, args.manifest)
    root = control_root(repo_root, inventory)
    ready_path = root / "PREFLIGHT_READY.json"
    verification_path = root / "checkpoint_verification.json"
    ready = load_json(ready_path)
    if ready.get("status") != "ready":
        raise RuntimeError(f"preflight is not ready: {ready_path}")
    if ready.get("axolotl") != inventory["axolotl"]:
        raise RuntimeError("preflight Axolotl provenance is stale")
    if ready.get("runtime") != inventory["runtime"]:
        raise RuntimeError("preflight runtime provenance is stale")
    if ready.get("checkpoint_verification_sha256") != sha256_file(
        verification_path
    ):
        raise RuntimeError("preflight checkpoint verification is stale")
    verification = load_json(verification_path)
    if verification.get("full_hash_verification") is not True:
        raise RuntimeError("preflight did not fully hash checkpoints")
    print(f"AXOLOTL_PREFLIGHT_VERIFIED status={ready_path}", flush=True)


def record_submission(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_inventory(repo_root, args.manifest)
    output = control_root(repo_root, inventory) / "submission.json"
    if output.exists():
        raise FileExistsError(
            f"submission journal already exists; refusing duplicate submit: {output}"
        )

    def assignments(values: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for value in values:
            name, separator, job_id = value.partition("=")
            if separator != "=" or not name or not job_id or name in result:
                raise ValueError(f"invalid job assignment: {value}")
            result[name] = job_id
        return result

    arrays = assignments(args.array_job)
    merges = assignments(args.merge_job)
    if set(arrays) != EXPECTED_VARIANTS or set(merges) != EXPECTED_VARIANTS:
        raise RuntimeError("submission journal does not cover all eleven variants")
    value = {
        "artifact_schema_version": 1,
        "array_jobs": arrays,
        "audit_job": args.audit_job,
        "checkpoint_count": 11,
        "eval_task_count": 44,
        "merge_jobs": merges,
        "preflight_job": args.preflight_job,
        "slurm": {
            "account": "raivn-ckpt",
            "array": "0-3",
            "mail_type": "END,FAIL",
            "mail_user": "suryadv@cs.washington.edu",
            "partition": "ckpt-all",
            "qos": "ckpt",
            "throttle": None,
        },
        "submitted_at": now_iso(),
    }
    atomic_text(output, canonical_json(value))
    print(f"AXOLOTL_SUBMISSION_RECORDED journal={output}", flush=True)


def print_path(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_inventory(repo_root, args.manifest)
    if args.field in {"eval_root", "control_root"}:
        path = (
            eval_root(repo_root, inventory)
            if args.field == "eval_root"
            else control_root(repo_root, inventory)
        )
    elif args.field == "benchmark":
        path = benchmark_paths(repo_root, inventory)[0]
    elif args.field == "shard":
        if args.shard_index is None:
            raise RuntimeError("--shard-index is required for field=shard")
        path = benchmark_paths(repo_root, inventory)[1][args.shard_index]
    else:
        if not args.variant:
            raise RuntimeError("--variant is required for checkpoint fields")
        item = item_for_variant(inventory, args.variant)
        path = absolute_item_paths(repo_root, item)[args.field]
    print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source")
    prepare_parser.set_defaults(handler=prepare)

    verify_parser = subparsers.add_parser("verify-checkpoints")
    verify_parser.add_argument("--full", action="store_true")
    verify_parser.add_argument("--status")
    verify_parser.set_defaults(handler=verify_checkpoints)

    repair_parser = subparsers.add_parser("repair-shard")
    repair_parser.add_argument("--eval-file", required=True)
    repair_parser.add_argument("--output-dir", required=True)
    repair_parser.set_defaults(handler=repair_shard)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--variant", required=True)
    merge_parser.set_defaults(handler=merge)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.set_defaults(handler=audit)

    preflight_parser = subparsers.add_parser("mark-preflight")
    preflight_parser.add_argument("--job-id", required=True)
    preflight_parser.set_defaults(handler=mark_preflight)

    check_preflight_parser = subparsers.add_parser("check-preflight")
    check_preflight_parser.set_defaults(handler=check_preflight)

    record_parser = subparsers.add_parser("record-submission")
    record_parser.add_argument("--preflight-job", required=True)
    record_parser.add_argument("--array-job", action="append", default=[])
    record_parser.add_argument("--merge-job", action="append", default=[])
    record_parser.add_argument("--audit-job", required=True)
    record_parser.set_defaults(handler=record_submission)

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
            "benchmark",
            "shard",
        ],
        required=True,
    )
    path_parser.add_argument("--variant")
    path_parser.add_argument("--shard-index", type=int, choices=range(4))
    path_parser.set_defaults(handler=print_path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
