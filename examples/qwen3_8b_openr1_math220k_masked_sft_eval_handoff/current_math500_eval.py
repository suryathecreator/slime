#!/usr/bin/env python3
"""Control, merge, and SLIME-V4 rescore the base/40K MATH-500 evaluations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from . import checkpoint_manifest as checkpoint_manifests
except ImportError:
    import checkpoint_manifest as checkpoint_manifests

from examples.qwen3_8b_32b_full_repro.math500_scorer_v4 import (
    SCORER_VERSION,
    score_response,
)


HANDOFF = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HANDOFF / "base_40k_eval_manifest.json"
TWO_K_MANIFEST = HANDOFF / "available_eval_manifest.json"
EXPECTED_TOTAL = 500
EXPECTED_CURRENT_IDS = {
    "base_8b",
    "40k_correct_only",
    "40k_weighted_tau_0p20",
    "40k_unmasked",
}
RESCORE_DIRNAME = Path("rescored") / SCORER_VERSION


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def canonical_row(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
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


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_current_inventory(
    repo_root: Path, manifest_path: str | Path = DEFAULT_MANIFEST
) -> dict[str, Any]:
    path = resolve_repo_path(repo_root, manifest_path)
    inventory = load_json(path)
    items = inventory.get("checkpoints")
    if not isinstance(items, list) or len(items) != 4:
        raise RuntimeError("base/40K manifest must contain four checkpoints")
    ids = [str(item["id"]) for item in items]
    if len(set(ids)) != 4 or set(ids) != EXPECTED_CURRENT_IDS:
        raise RuntimeError(f"unexpected base/40K checkpoint IDs: {ids}")
    if inventory.get("runtime") != {
        "math_verify": "0.9.0",
        "torch": "2.8.0",
        "transformers": "4.57.6",
        "vllm": "0.10.2",
    }:
        raise RuntimeError("base/40K runtime pin changed")
    if inventory.get("scorer", {}).get("version") != SCORER_VERSION:
        raise RuntimeError("base/40K scorer pin changed")
    return inventory


def load_2k_inventory(repo_root: Path) -> dict[str, Any]:
    inventory = load_json(resolve_repo_path(repo_root, TWO_K_MANIFEST))
    items = inventory.get("checkpoints")
    if not isinstance(items, list) or len(items) != 11:
        raise RuntimeError("2K manifest must contain eleven checkpoints")
    ids = [str(item["id"]) for item in items]
    if len(set(ids)) != 11 or any(not item.startswith("2k_") for item in ids):
        raise RuntimeError("2K checkpoint IDs are not unique/pinned")
    return inventory


def item_for_id(inventory: dict[str, Any], eval_id: str) -> dict[str, Any]:
    matches = [
        item for item in inventory["checkpoints"] if item["id"] == eval_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate evaluation ID: {eval_id}")
    return matches[0]


def item_paths(
    repo_root: Path, item: dict[str, Any]
) -> dict[str, Path]:
    return {
        key: resolve_repo_path(repo_root, item[key])
        for key in ("manifest", "model", "output", "result")
    }


def control_root(repo_root: Path, inventory: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, inventory["control_root"])


def benchmark_paths(
    repo_root: Path, inventory: dict[str, Any]
) -> tuple[Path, list[Path], Path]:
    benchmark = inventory["benchmark"]
    return (
        resolve_repo_path(repo_root, benchmark["full"]),
        [resolve_repo_path(repo_root, value) for value in benchmark["shards"]],
        resolve_repo_path(repo_root, benchmark["smoke"]),
    )


def require_generation_policy(
    repo_root: Path, inventory: dict[str, Any]
) -> dict[str, Any]:
    policy = load_json(
        resolve_repo_path(repo_root, inventory["generation_policy"])
    )
    expected = {
        "max_tokens": 32768,
        "n": 1,
        "stop_token_ids": [151643, 151645],
        "temperature": 0.0,
    }
    for key, value in expected.items():
        if policy.get("decoding", {}).get(key) != value:
            raise RuntimeError(f"generation policy changed: decoding.{key}")
    engine = policy.get("engine", {})
    expected_engine = {
        "dtype": "bfloat16",
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "gpu_memory_utilization": 0.92,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "tensor_parallel_size": 1,
        "vllm_version": "0.10.2",
    }
    for key, value in expected_engine.items():
        if engine.get(key) != value:
            raise RuntimeError(f"generation policy changed: engine.{key}")
    if policy.get("persistence", {}).get("chunk_size") != 32:
        raise RuntimeError("generation chunk size is not 32")
    if policy.get("persistence", {}).get("layout") != (
        "four_contiguous_125_problem_shards"
    ):
        raise RuntimeError("generation sharding policy changed")
    if policy.get("prompt", {}).get("user_content") != (
        'instruction + "\\n\\nProblem:\\n" + problem'
    ):
        raise RuntimeError("current evaluation prompt order changed")
    return policy


def prepare(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    policy = require_generation_policy(repo_root, inventory)
    full, shards, smoke = benchmark_paths(repo_root, inventory)
    if sha256_file(full) != inventory["benchmark"]["sha256"]:
        raise RuntimeError("canonical MATH-500 JSONL hash changed")
    rows = read_jsonl(full)
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError("canonical MATH-500 JSONL is not 500 rows")
    if len(shards) != 4 or any(len(read_jsonl(path)) != 125 for path in shards):
        raise RuntimeError("MATH-500 shard layout is not four by 125")
    if len(read_jsonl(smoke)) != 1:
        raise RuntimeError("MATH-500 smoke input is not one row")
    tokenizer = resolve_repo_path(repo_root, inventory["tokenizer"])
    provenance = tokenizer / "TOKENIZER_PROVENANCE.json"
    for name in (
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "TOKENIZER_PROVENANCE.json",
    ):
        if not (tokenizer / name).is_file():
            raise FileNotFoundError(tokenizer / name)
    value = {
        "artifact_schema_version": 1,
        "benchmark_sha256": sha256_file(full),
        "generation_policy_sha256": sha256_file(
            resolve_repo_path(repo_root, inventory["generation_policy"])
        ),
        "prepared_at": now_iso(),
        "scorer_version": SCORER_VERSION,
        "status": "prepared",
        "tokenizer_provenance_sha256": sha256_file(provenance),
        "vllm_version": policy["engine"]["vllm_version"],
    }
    output = control_root(repo_root, inventory) / "PREPARED.json"
    atomic_text(output, canonical_json(value))
    print(f"CURRENT_EVAL_PREPARED status={output}", flush=True)


def quick_checkpoint_check(checkpoint: Path, manifest_path: Path) -> str:
    manifest = load_json(manifest_path)
    checkpoint_manifests.validate_manifest(manifest)
    expected = {
        item["name"]: int(item["bytes"]) for item in manifest["files"]
    }
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
            f"checkpoint inventory mismatch at {checkpoint}: "
            f"missing={missing} extra={extra} changed={changed}"
        )
    return str(manifest["checkpoint_manifest_sha256"])


def verify_checkpoints(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    identities: dict[str, str] = {}
    for item in inventory["checkpoints"]:
        paths = item_paths(repo_root, item)
        identity = (
            checkpoint_manifests.verify(paths["manifest"], paths["model"])
            if args.full
            else quick_checkpoint_check(paths["model"], paths["manifest"])
        )
        manifest = load_json(paths["manifest"])
        for key in ("family", "final_iteration", "full_sft", "variant"):
            if manifest.get(key) != item.get(key):
                raise RuntimeError(
                    f"{item['id']} checkpoint manifest {key} mismatch"
                )
        identities[str(item["id"])] = identity
        print(
            f"CURRENT_CHECKPOINT_VERIFIED id={item['id']} "
            f"full={int(args.full)} sha256={identity}",
            flush=True,
        )
    if args.status:
        atomic_text(
            Path(args.status),
            canonical_json(
                {
                    "artifact_schema_version": 1,
                    "checked_at": now_iso(),
                    "full_hash_verification": bool(args.full),
                    "identities": identities,
                }
            ),
        )


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
    expected_rows = read_jsonl(Path(args.eval_file))
    expected_ids = [str(row["problem_id"]) for row in expected_rows]
    if len(expected_ids) != 125 or len(set(expected_ids)) != 125:
        raise RuntimeError("eval shard must contain 125 unique problem IDs")
    expected_set = set(expected_ids)
    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions.jsonl"
    raw_path = output_dir / "raw_generations.jsonl"
    predictions = dedupe_latest(
        read_jsonl(prediction_path) if prediction_path.is_file() else [],
        expected_set,
    )
    raw = dedupe_latest(
        read_jsonl(raw_path) if raw_path.is_file() else [], expected_set
    )
    common = set(predictions) & set(raw)
    if prediction_path.exists() or common:
        atomic_text(
            prediction_path,
            jsonl_text(
                predictions[item] for item in expected_ids if item in common
            ),
        )
    if raw_path.exists() or common:
        atomic_text(
            raw_path,
            jsonl_text(raw[item] for item in expected_ids if item in common),
        )
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and common != expected_set:
        raise RuntimeError("completed shard has incomplete source JSONL")
    print(
        f"CURRENT_SHARD_REPAIRED output={output_dir} "
        f"completed={len(common)} expected=125",
        flush=True,
    )


def record_invocation(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    item = item_for_id(inventory, args.eval_id)
    paths = item_paths(repo_root, item)
    _, shards, _ = benchmark_paths(repo_root, inventory)
    if args.shard_index not in range(4):
        raise RuntimeError("shard index must be 0..3")
    generation_policy_path = resolve_repo_path(
        repo_root, inventory["generation_policy"]
    )
    tokenizer = resolve_repo_path(repo_root, inventory["tokenizer"])
    checkpoint = load_json(paths["manifest"])
    value = {
        "artifact_schema_version": 1,
        "checkpoint_manifest_sha256": checkpoint[
            "checkpoint_manifest_sha256"
        ],
        "eval_id": item["id"],
        "evaluator": inventory["axolotl"],
        "generation_policy_sha256": sha256_file(generation_policy_path),
        "invocation": {
            "chunk_size": 32,
            "gpu_memory_utilization": 0.92,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "max_tokens": 32768,
            "repeat": 0,
            "sampling_mode": "current_2k_matched",
            "train_seed": 42,
        },
        "model": str(paths["model"]),
        "shard_index": args.shard_index,
        "source_eval_file": str(shards[args.shard_index]),
        "source_eval_file_sha256": sha256_file(shards[args.shard_index]),
        "tokenizer": str(tokenizer),
        "tokenizer_provenance_sha256": sha256_file(
            tokenizer / "TOKENIZER_PROVENANCE.json"
        ),
    }
    output = (
        paths["output"]
        / "shards"
        / f"shard_{args.shard_index:02d}"
        / "INVOCATION.json"
    )
    content = canonical_json(value)
    if output.is_file() and output.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"refusing incompatible shard invocation: {output}")
    if not output.exists():
        atomic_text(output, content)
    print(
        f"CURRENT_INVOCATION_RECORDED id={item['id']} "
        f"shard={args.shard_index} path={output}",
        flush=True,
    )


def validate_source_rows(
    canonical_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(canonical_rows) != len(predictions) or len(predictions) != len(raw_rows):
        raise RuntimeError(
            f"source row count mismatch at {source}: "
            f"canonical={len(canonical_rows)} predictions={len(predictions)} "
            f"raw={len(raw_rows)}"
        )
    expected_ids = [str(row["problem_id"]) for row in canonical_rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise RuntimeError(f"canonical source has duplicate IDs: {source}")
    pred_by_id = dedupe_latest(predictions, set(expected_ids))
    raw_by_id = dedupe_latest(raw_rows, set(expected_ids))
    if set(pred_by_id) != set(expected_ids) or set(raw_by_id) != set(
        expected_ids
    ):
        raise RuntimeError(f"source IDs are missing or duplicated: {source}")
    ordered_predictions = [pred_by_id[item] for item in expected_ids]
    ordered_raw = [raw_by_id[item] for item in expected_ids]
    for canonical, prediction, raw in zip(
        canonical_rows, ordered_predictions, ordered_raw, strict=True
    ):
        problem_id = str(canonical["problem_id"])
        if str(prediction["problem_id"]) != problem_id:
            raise RuntimeError(f"prediction order mismatch: {problem_id}")
        if str(raw["problem_id"]) != problem_id:
            raise RuntimeError(f"raw generation order mismatch: {problem_id}")
        if str(prediction["correct_answer"]) != str(
            canonical["correct_answer"]
        ):
            raise RuntimeError(f"gold answer mismatch: {problem_id}")
        if prediction.get("finish_reason") != raw.get("finish_reason"):
            raise RuntimeError(f"finish reason mismatch: {problem_id}")
        if not isinstance(raw.get("generation"), str):
            raise RuntimeError(f"missing raw generation: {problem_id}")
    return ordered_predictions, ordered_raw


def merge_current(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    item = item_for_id(inventory, args.eval_id)
    paths = item_paths(repo_root, item)
    full, shard_files, _ = benchmark_paths(repo_root, inventory)
    canonical_rows = read_jsonl(full)
    predictions: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    native_metrics: list[dict[str, Any]] = []
    for index, shard_file in enumerate(shard_files):
        shard_dir = paths["output"] / "shards" / f"shard_{index:02d}"
        invocation = load_json(shard_dir / "INVOCATION.json")
        if invocation.get("eval_id") != item["id"]:
            raise RuntimeError(f"shard invocation ID mismatch: {shard_dir}")
        if int(invocation.get("shard_index", -1)) != index:
            raise RuntimeError(
                f"shard invocation index mismatch: {shard_dir}"
            )
        if invocation.get("invocation") != {
            "chunk_size": 32,
            "gpu_memory_utilization": 0.92,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "max_tokens": 32768,
            "repeat": 0,
            "sampling_mode": "current_2k_matched",
            "train_seed": 42,
        }:
            raise RuntimeError(f"shard invocation policy mismatch: {shard_dir}")
        shard_canonical = read_jsonl(shard_file)
        shard_predictions, shard_raw = validate_source_rows(
            shard_canonical,
            read_jsonl(shard_dir / "predictions.jsonl"),
            read_jsonl(shard_dir / "raw_generations.jsonl"),
            source=str(shard_dir),
        )
        metrics = load_json(shard_dir / "metrics.json")
        if int(metrics.get("num_eval_problems", -1)) != 125:
            raise RuntimeError(f"incomplete native metrics: {shard_dir}")
        if metrics.get("variant") != item["id"]:
            raise RuntimeError(f"native metrics variant mismatch: {shard_dir}")
        if metrics.get("sampling_mode") != "current_2k_matched":
            raise RuntimeError(
                f"native metrics sampling mode mismatch: {shard_dir}"
            )
        if int(metrics.get("train_seed", -1)) != 42:
            raise RuntimeError(f"native metrics seed mismatch: {shard_dir}")
        if int(metrics.get("repeat", -1)) != 0:
            raise RuntimeError(f"native metrics repeat mismatch: {shard_dir}")
        if metrics.get("load_mode") != "full_sft":
            raise RuntimeError(f"native metrics load mode mismatch: {shard_dir}")
        if Path(str(metrics.get("eval_file"))).resolve() != shard_file:
            raise RuntimeError(f"native metrics eval file mismatch: {shard_dir}")
        if any(row.get("variant") != item["id"] for row in shard_predictions):
            raise RuntimeError(f"prediction variant mismatch: {shard_dir}")
        if any(row.get("variant") != item["id"] for row in shard_raw):
            raise RuntimeError(f"raw generation variant mismatch: {shard_dir}")
        predictions.extend(shard_predictions)
        raw_rows.extend(shard_raw)
        native_metrics.append(metrics)
    predictions, raw_rows = validate_source_rows(
        canonical_rows, predictions, raw_rows, source=str(paths["output"])
    )
    merged = paths["output"] / "merged"
    atomic_text(merged / "predictions.jsonl", jsonl_text(predictions))
    atomic_text(merged / "raw_generations.jsonl", jsonl_text(raw_rows))
    correct = sum(bool(row["is_correct"]) for row in predictions)
    summary = {
        "artifact_schema_version": 1,
        "accuracy": correct / EXPECTED_TOTAL,
        "benchmark": "HuggingFaceH4/MATH-500",
        "cap_hits": sum(bool(row["cap_hit"]) for row in predictions),
        "correct": correct,
        "dataset_revision": inventory["dataset"]["revision"],
        "evaluator": (
            "Axolotl-Masked-SFT/scripts/openr1_axolotl/"
            "evaluate_math_vllm.py"
        ),
        "finish_reasons": dict(
            Counter(str(row.get("finish_reason")) for row in predictions)
        ),
        "native_shard_metrics": native_metrics,
        "stage": item["id"],
        "total": EXPECTED_TOTAL,
        "variant": item["variant"],
    }
    atomic_text(merged / "summary.json", canonical_json(summary))
    atomic_text(
        merged / "summary.sha256",
        sha256_file(merged / "summary.json") + "\n",
    )
    print(
        f"CURRENT_GENERATIONS_MERGED id={item['id']} "
        f"native_correct={correct}/500 output={merged}",
        flush=True,
    )


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    correct: list[bool], seed: int, samples: int = 10_000
) -> list[float]:
    rng = random.Random(seed)
    count = len(correct)
    values = [
        sum(correct[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    values.sort()
    return [
        values[int(0.025 * (samples - 1))],
        values[int(0.975 * (samples - 1))],
    ]


def binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    probability = sum(
        math.comb(n, index) for index in range(0, min(k, n - k) + 1)
    ) / (2**n)
    return min(1.0, 2 * probability)


def compact_result_files(
    summary: dict[str, Any],
    checkpoint: dict[str, Any],
    generation_policy: dict[str, Any],
    rescore_policy: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, str]:
    markdown = (
        f"# {summary['stage']} MATH-500 — SLIME scorer V4\n\n"
        f"- SLIME V4: {summary['correct']}/{summary['total']} "
        f"({summary['pass_at_1']:.2%})\n"
        f"- Original Axolotl score: {summary['native_correct']}/"
        f"{summary['total']}\n"
        f"- Cap hits marked wrong: {summary['cap_hits']}\n"
        f"- Scorer: `{summary['scorer_version']}`\n"
        f"- Checkpoint: `{summary['checkpoint_manifest_sha256']}`\n"
    )
    return {
        "CHECKPOINT.json": canonical_json(checkpoint),
        "GENERATION_POLICY.json": canonical_json(generation_policy),
        "PROVENANCE.json": canonical_json(provenance),
        "RESCORE_POLICY.json": canonical_json(rescore_policy),
        "RESULTS.json": canonical_json(summary),
        "RESULTS.md": markdown,
    }


def write_compact_result(
    destination: Path, expected: dict[str, str]
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    allowed = set(expected)
    unexpected = sorted(
        path.name
        for path in destination.iterdir()
        if path.is_file() and path.name not in allowed
    )
    if unexpected:
        raise RuntimeError(
            f"unexpected compact rescore files at {destination}: {unexpected}"
        )
    for name, content in expected.items():
        path = destination / name
        if path.is_file() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to replace incompatible result: {path}")
        if not path.exists():
            atomic_text(path, content)


def rescore_item(
    *,
    repo_root: Path,
    current_inventory: dict[str, Any],
    item: dict[str, Any],
    bootstrap_seed: int,
) -> dict[str, Any]:
    paths = item_paths(repo_root, item)
    full, _, _ = benchmark_paths(repo_root, current_inventory)
    canonical_rows = read_jsonl(full)
    merged = paths["output"] / "merged"
    prediction_path = merged / "predictions.jsonl"
    raw_path = merged / "raw_generations.jsonl"
    predictions, raw_rows = validate_source_rows(
        canonical_rows,
        read_jsonl(prediction_path),
        read_jsonl(raw_path),
        source=str(merged),
    )
    checkpoint = load_json(paths["manifest"])
    checkpoint_manifests.validate_manifest(checkpoint)
    generation_policy_path = resolve_repo_path(
        repo_root, current_inventory["generation_policy"]
    )
    generation_policy = require_generation_policy(
        repo_root, current_inventory
    )
    rescore_policy_path = resolve_repo_path(
        repo_root, current_inventory["rescore_policy"]
    )
    rescore_policy = load_json(rescore_policy_path)
    output = paths["output"] / RESCORE_DIRNAME
    problem_output = output / "problems"
    problem_output.mkdir(parents=True, exist_ok=True)
    compact_rows: list[dict[str, Any]] = []
    trace_artifacts: list[dict[str, Any]] = []
    for index, (canonical, prediction, raw) in enumerate(
        zip(canonical_rows, predictions, raw_rows, strict=True)
    ):
        problem_id = str(canonical["problem_id"])
        generation = str(raw["generation"])
        trace = score_response(
            generation,
            str(prediction["correct_answer"]),
            finish_reason=prediction.get("finish_reason"),
            stop_token_id=None,
            cap_hit=bool(prediction["cap_hit"]),
            problem_id=problem_id,
        )
        if trace.get("scorer_version") != SCORER_VERSION:
            raise RuntimeError(f"scorer version mismatch: {problem_id}")
        result = {
            "artifact_schema_version": "saved_generation_slime_v4_rescore_v1",
            "cap_hit": bool(prediction["cap_hit"]),
            "correct_answer": str(prediction["correct_answer"]),
            "eval_index": index,
            "finish_reason": prediction.get("finish_reason"),
            "generated_token_count": int(prediction["generated_token_count"]),
            "problem_id": problem_id,
            "scorer_trace": trace,
            "source": {
                "generation_sha256": sha256_bytes(generation.encode()),
                "prediction_record_sha256": sha256_bytes(
                    canonical_row(prediction)
                ),
                "raw_record_sha256": sha256_bytes(canonical_row(raw)),
                "stop_token_id": None,
            },
            "source_axolotl": {
                "candidate": prediction.get("predicted_answer"),
                "is_correct": bool(prediction["is_correct"]),
            },
            "stage": item["id"],
        }
        target = problem_output / f"problem_{index:04d}.json"
        atomic_text(target, canonical_json(result))
        trace_artifacts.append(
            {
                "bytes": target.stat().st_size,
                "eval_index": index,
                "path": str(target),
                "sha256": sha256_file(target),
            }
        )
        compact_rows.append(
            {
                "candidate": trace["official_extracted_candidate"],
                "candidate_sha256": trace["official_candidate_sha256"],
                "cap_hit": bool(prediction["cap_hit"]),
                "eval_index": index,
                "generated_text_sha256": trace["generated_text_sha256"],
                "is_correct": bool(trace["is_correct"]),
                "native_candidate": prediction.get("predicted_answer"),
                "native_is_correct": bool(prediction["is_correct"]),
                "official_decision_reason": trace[
                    "official_decision_reason"
                ],
                "problem_id": problem_id,
                "think_tag_state": trace["think_tag_state"],
            }
        )
    correct = [bool(row["is_correct"]) for row in compact_rows]
    native = [bool(row["native_is_correct"]) for row in compact_rows]
    wrong_to_correct = sum(
        not before and after for before, after in zip(native, correct, strict=True)
    )
    correct_to_wrong = sum(
        before and not after for before, after in zip(native, correct, strict=True)
    )
    lengths = [int(row["generated_token_count"]) for row in predictions]
    summary = {
        "artifact_schema_version": 1,
        "bootstrap_95_ci": bootstrap_ci(correct, bootstrap_seed),
        "cap_hits": sum(bool(row["cap_hit"]) for row in compact_rows),
        "checkpoint_manifest_sha256": checkpoint[
            "checkpoint_manifest_sha256"
        ],
        "correct": sum(correct),
        "eligible_final_answers": sum(
            row["candidate"] is not None and not row["cap_hit"]
            for row in compact_rows
        ),
        "finish_reasons": dict(
            Counter(str(row.get("finish_reason")) for row in predictions)
        ),
        "length": {
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
        },
        "native_correct": sum(native),
        "native_to_v4": {
            "correct_to_incorrect": correct_to_wrong,
            "incorrect_to_correct": wrong_to_correct,
            "mcnemar_exact_p": binomial_two_sided(
                min(wrong_to_correct, correct_to_wrong),
                wrong_to_correct + correct_to_wrong,
            ),
        },
        "parseable": sum(row["candidate"] is not None for row in compact_rows),
        "pass_at_1": sum(correct) / EXPECTED_TOTAL,
        "per_problem": compact_rows,
        "scorer_version": SCORER_VERSION,
        "source_generation_policy_sha256": sha256_file(
            generation_policy_path
        ),
        "source_predictions_sha256": sha256_file(prediction_path),
        "source_raw_generations_sha256": sha256_file(raw_path),
        "stage": item["id"],
        "think_tag_states": dict(
            Counter(row["think_tag_state"] for row in compact_rows)
        ),
        "total": EXPECTED_TOTAL,
        "trace_artifact_manifest_sha256": sha256_bytes(
            json.dumps(
                trace_artifacts, sort_keys=True, separators=(",", ":")
            ).encode()
        ),
        "variant": item["variant"],
    }
    atomic_text(output / "summary.json", canonical_json(summary))
    atomic_text(output / "per_problem.jsonl", jsonl_text(compact_rows))
    atomic_text(
        output / "summary.md",
        compact_result_files(
            summary, checkpoint, generation_policy, rescore_policy, {}
        )["RESULTS.md"],
    )
    provenance = {
        "artifact_schema_version": 1,
        "checkpoint_manifest": str(paths["manifest"]),
        "checkpoint_manifest_sha256": checkpoint[
            "checkpoint_manifest_sha256"
        ],
        "generation_policy_sha256": sha256_file(generation_policy_path),
        "raw_results": str(output),
        "rescore_policy_sha256": sha256_file(rescore_policy_path),
        "scorer_module_sha256": sha256_file(
            repo_root / current_inventory["scorer"]["module"]
        ),
        "source_predictions": str(prediction_path),
        "source_predictions_sha256": sha256_file(prediction_path),
        "source_raw_generations": str(raw_path),
        "source_raw_generations_sha256": sha256_file(raw_path),
        "stage": item["id"],
        "stop_token_id": None,
        "stop_token_note": (
            "The source Axolotl artifact did not retain the exact stopping "
            "token. This diagnostic field does not affect V4 correctness."
        ),
    }
    atomic_text(output / "provenance.json", canonical_json(provenance))
    write_compact_result(
        paths["result"] / RESCORE_DIRNAME,
        compact_result_files(
            summary,
            checkpoint,
            generation_policy,
            rescore_policy,
            provenance,
        ),
    )
    print(
        f"SLIME_V4_RESCORE_COMPLETE id={item['id']} "
        f"native={sum(native)}/500 v4={sum(correct)}/500 output={output}",
        flush=True,
    )
    return summary


def rescore_one(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    item = item_for_id(inventory, args.eval_id)
    rescore_item(
        repo_root=repo_root,
        current_inventory=inventory,
        item=item,
        bootstrap_seed=args.bootstrap_seed,
    )


def rescore_2k(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    current = load_current_inventory(repo_root, args.manifest)
    inventory = load_2k_inventory(repo_root)
    for index, item in enumerate(inventory["checkpoints"]):
        rescore_item(
            repo_root=repo_root,
            current_inventory=current,
            item=item,
            bootstrap_seed=args.bootstrap_seed + index,
        )
    print("SLIME_V4_2K_RESCORE_COMPLETE checkpoints=11 rows=5500", flush=True)


def rescore_2k_one(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    current = load_current_inventory(repo_root, args.manifest)
    inventory = load_2k_inventory(repo_root)
    item = item_for_id(inventory, args.eval_id)
    rescore_item(
        repo_root=repo_root,
        current_inventory=current,
        item=item,
        bootstrap_seed=args.bootstrap_seed,
    )


def paired_against_base(
    base_rows: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(base_rows) != EXPECTED_TOTAL or len(rows) != EXPECTED_TOTAL:
        raise RuntimeError("paired comparison requires two 500-row results")
    if [row["eval_index"] for row in base_rows] != list(range(EXPECTED_TOTAL)):
        raise RuntimeError("base result order changed")
    if [row["eval_index"] for row in rows] != list(range(EXPECTED_TOTAL)):
        raise RuntimeError("comparison result order changed")
    base_correct = [bool(row["is_correct"]) for row in base_rows]
    current_correct = [bool(row["is_correct"]) for row in rows]
    wrong_to_correct = sum(
        not before and after
        for before, after in zip(base_correct, current_correct, strict=True)
    )
    correct_to_wrong = sum(
        before and not after
        for before, after in zip(base_correct, current_correct, strict=True)
    )
    return {
        "base_correct_to_current_incorrect": correct_to_wrong,
        "base_incorrect_to_current_correct": wrong_to_correct,
        "mcnemar_exact_p": binomial_two_sided(
            min(wrong_to_correct, correct_to_wrong),
            wrong_to_correct + correct_to_wrong,
        ),
    }


def rescore_summary_path(repo_root: Path, item: dict[str, Any]) -> Path:
    return item_paths(repo_root, item)["output"] / RESCORE_DIRNAME / "summary.json"


def build_report(repo_root: Path, current: dict[str, Any]) -> dict[str, Any]:
    two_k = load_2k_inventory(repo_root)
    current_items = {
        str(item["id"]): item for item in current["checkpoints"]
    }
    base_summary = load_json(
        rescore_summary_path(repo_root, current_items["base_8b"])
    )
    base_rows = list(base_summary["per_problem"])
    report_rows: list[dict[str, Any]] = []
    ordered: list[tuple[str, dict[str, Any]]] = [
        ("base", current_items["base_8b"])
    ]
    ordered.extend(("2k", item) for item in two_k["checkpoints"])
    ordered.extend(
        ("40k", current_items[eval_id])
        for eval_id in (
            "40k_correct_only",
            "40k_weighted_tau_0p20",
            "40k_unmasked",
        )
    )
    for family, item in ordered:
        summary = load_json(rescore_summary_path(repo_root, item))
        if (
            int(summary.get("total", -1)) != EXPECTED_TOTAL
            or summary.get("scorer_version") != SCORER_VERSION
        ):
            raise RuntimeError(f"incomplete V4 result: {item['id']}")
        paired = (
            {
                "base_correct_to_current_incorrect": 0,
                "base_incorrect_to_current_correct": 0,
                "mcnemar_exact_p": 1.0,
            }
            if item["id"] == "base_8b"
            else paired_against_base(base_rows, list(summary["per_problem"]))
        )
        report_rows.append(
            {
                "checkpoint_manifest_sha256": summary[
                    "checkpoint_manifest_sha256"
                ],
                "correct": summary["correct"],
                "family": family,
                "native_correct": summary["native_correct"],
                "paired_against_base": paired,
                "pass_at_1": summary["pass_at_1"],
                "stage": item["id"],
                "total": EXPECTED_TOTAL,
            }
        )
    report = {
        "artifact_schema_version": 1,
        "base_stage": "base_8b",
        "generated_at": now_iso(),
        "result_count": len(report_rows),
        "results": report_rows,
        "scorer_version": SCORER_VERSION,
        "status": "complete",
        "total_rescored_rows": len(report_rows) * EXPECTED_TOTAL,
    }
    report_json = resolve_repo_path(repo_root, current["report_json"])
    report_md = resolve_repo_path(repo_root, current["report_md"])
    atomic_text(report_json, canonical_json(report))
    lines = [
        "# MATH-500 results with the shared SLIME V4 scorer",
        "",
        "All rows below use `math500_strict_boxed_units_scorer_v4`. Existing 2K "
        "generations were rescored without inference; base and 40K generations "
        "use the same generation policy as the completed 2K evaluations.",
        "",
        "| Family | Stage | Native | SLIME V4 | Base wrong→correct | Base correct→wrong |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        paired = row["paired_against_base"]
        lines.append(
            f"| {row['family']} | `{row['stage']}` | "
            f"{row['native_correct']}/500 | {row['correct']}/500 "
            f"({row['pass_at_1']:.2%}) | "
            f"{paired['base_incorrect_to_current_correct']} | "
            f"{paired['base_correct_to_current_incorrect']} |"
        )
    atomic_text(report_md, "\n".join(lines) + "\n")
    return report


def audit_current(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    for item in inventory["checkpoints"]:
        summary = load_json(rescore_summary_path(repo_root, item))
        if int(summary.get("total", -1)) != EXPECTED_TOTAL:
            raise RuntimeError(f"incomplete current result: {item['id']}")
        if summary.get("scorer_version") != SCORER_VERSION:
            raise RuntimeError(f"wrong current scorer: {item['id']}")
        if len(summary.get("per_problem", [])) != EXPECTED_TOTAL:
            raise RuntimeError(f"incomplete current decisions: {item['id']}")
    report = build_report(repo_root, inventory)
    audit = {
        "artifact_schema_version": 1,
        "audited_at": now_iso(),
        "current_checkpoint_count": 4,
        "current_row_count": 2000,
        "report_sha256": sha256_file(
            resolve_repo_path(repo_root, inventory["report_json"])
        ),
        "scorer_version": SCORER_VERSION,
        "status": report["status"],
        "total_comparable_results": report["result_count"],
        "total_rescored_rows": report["total_rescored_rows"],
    }
    output = control_root(repo_root, inventory) / "FINAL_V4_AUDIT.json"
    atomic_text(output, canonical_json(audit))
    print(f"CURRENT_V4_AUDIT_COMPLETE audit={output}", flush=True)


def runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(distribution)
        for name, distribution in (
            ("vllm", "vllm"),
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("math_verify", "math-verify"),
        )
    }


def mark_ready(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    root = control_root(repo_root, inventory)
    prepared = root / "PREPARED.json"
    verification = root / "checkpoint_verification.json"
    if not prepared.is_file() or not verification.is_file():
        raise RuntimeError("preflight preparation or verification is missing")
    verified = load_json(verification)
    if verified.get("full_hash_verification") is not True:
        raise RuntimeError("preflight did not fully hash checkpoints")
    if set(verified.get("identities", {})) != EXPECTED_CURRENT_IDS:
        raise RuntimeError("preflight checkpoint identities are incomplete")
    versions = runtime_versions()
    if versions != inventory["runtime"]:
        raise RuntimeError(f"runtime mismatch: {versions} != {inventory['runtime']}")
    value = {
        "artifact_schema_version": 1,
        "checkpoint_verification_sha256": sha256_file(verification),
        "job_id": args.job_id,
        "prepared_sha256": sha256_file(prepared),
        "ready_at": now_iso(),
        "runtime": versions,
        "scorer_version": SCORER_VERSION,
        "status": "ready",
    }
    atomic_text(root / "PREFLIGHT_READY.json", canonical_json(value))
    print(f"CURRENT_PREFLIGHT_READY root={root}", flush=True)


def check_ready(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    root = control_root(repo_root, inventory)
    ready = load_json(root / "PREFLIGHT_READY.json")
    if ready.get("status") != "ready":
        raise RuntimeError("current evaluation preflight is not ready")
    if ready.get("runtime") != inventory["runtime"]:
        raise RuntimeError("current evaluation preflight runtime is stale")
    if ready.get("scorer_version") != SCORER_VERSION:
        raise RuntimeError("current evaluation preflight scorer is stale")
    if ready.get("prepared_sha256") != sha256_file(root / "PREPARED.json"):
        raise RuntimeError("current evaluation preparation changed")
    if ready.get("checkpoint_verification_sha256") != sha256_file(
        root / "checkpoint_verification.json"
    ):
        raise RuntimeError("current evaluation checkpoint verification changed")
    print(f"CURRENT_PREFLIGHT_VERIFIED root={root}", flush=True)


def record_submission(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    output = control_root(repo_root, inventory) / "submission.json"
    if output.exists():
        raise FileExistsError(f"submission already recorded: {output}")

    def assignments(values: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for value in values:
            name, separator, job_id = value.partition("=")
            if separator != "=" or name in result or not job_id.isdigit():
                raise RuntimeError(f"invalid job assignment: {value}")
            result[name] = job_id
        return result

    arrays = assignments(args.array_job)
    finalizers = assignments(args.finalize_job)
    if set(arrays) != EXPECTED_CURRENT_IDS or set(finalizers) != EXPECTED_CURRENT_IDS:
        raise RuntimeError("submission does not cover all four current checkpoints")
    value = {
        "artifact_schema_version": 1,
        "array_jobs": arrays,
        "audit_job": args.audit_job,
        "base_first_dependency": (
            "all 40K arrays depend afterok on the base_8b finalizer"
        ),
        "checkpoint_count": 4,
        "eval_task_count": 16,
        "finalize_jobs": finalizers,
        "preflight_job": args.preflight_job,
        "submitted_at": now_iso(),
    }
    atomic_text(output, canonical_json(value))
    print(f"CURRENT_SUBMISSION_RECORDED journal={output}", flush=True)


def print_path(args: argparse.Namespace) -> None:
    repo_root = repo_root_from_args(args)
    inventory = load_current_inventory(repo_root, args.manifest)
    if args.field == "control_root":
        path = control_root(repo_root, inventory)
    elif args.field == "tokenizer":
        path = resolve_repo_path(repo_root, inventory["tokenizer"])
    elif args.field in {"benchmark", "smoke"}:
        full, _, smoke = benchmark_paths(repo_root, inventory)
        path = full if args.field == "benchmark" else smoke
    elif args.field == "shard":
        if args.shard_index is None:
            raise RuntimeError("--shard-index is required for field=shard")
        path = benchmark_paths(repo_root, inventory)[1][args.shard_index]
    else:
        if not args.eval_id:
            raise RuntimeError("--eval-id is required for checkpoint fields")
        item = item_for_id(inventory, args.eval_id)
        path = item_paths(repo_root, item)[args.field]
    print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.set_defaults(handler=prepare)

    verify_parser = subparsers.add_parser("verify-checkpoints")
    verify_parser.add_argument("--full", action="store_true")
    verify_parser.add_argument("--status")
    verify_parser.set_defaults(handler=verify_checkpoints)

    repair_parser = subparsers.add_parser("repair-shard")
    repair_parser.add_argument("--eval-file", required=True)
    repair_parser.add_argument("--output-dir", required=True)
    repair_parser.set_defaults(handler=repair_shard)

    invocation_parser = subparsers.add_parser("record-invocation")
    invocation_parser.add_argument("--eval-id", required=True)
    invocation_parser.add_argument("--shard-index", type=int, required=True)
    invocation_parser.set_defaults(handler=record_invocation)

    merge_parser = subparsers.add_parser("merge-current")
    merge_parser.add_argument("--eval-id", required=True)
    merge_parser.set_defaults(handler=merge_current)

    rescore_parser = subparsers.add_parser("rescore-one")
    rescore_parser.add_argument("--eval-id", required=True)
    rescore_parser.add_argument("--bootstrap-seed", type=int, default=1234)
    rescore_parser.set_defaults(handler=rescore_one)

    rescore_2k_parser = subparsers.add_parser("rescore-2k")
    rescore_2k_parser.add_argument("--bootstrap-seed", type=int, default=1234)
    rescore_2k_parser.set_defaults(handler=rescore_2k)

    rescore_2k_one_parser = subparsers.add_parser("rescore-2k-one")
    rescore_2k_one_parser.add_argument("--eval-id", required=True)
    rescore_2k_one_parser.add_argument(
        "--bootstrap-seed", type=int, default=1234
    )
    rescore_2k_one_parser.set_defaults(handler=rescore_2k_one)

    audit_parser = subparsers.add_parser("audit-current")
    audit_parser.set_defaults(handler=audit_current)

    ready_parser = subparsers.add_parser("mark-ready")
    ready_parser.add_argument("--job-id", required=True)
    ready_parser.set_defaults(handler=mark_ready)

    check_parser = subparsers.add_parser("check-ready")
    check_parser.set_defaults(handler=check_ready)

    record_parser = subparsers.add_parser("record-submission")
    record_parser.add_argument("--preflight-job", required=True)
    record_parser.add_argument("--array-job", action="append", default=[])
    record_parser.add_argument("--finalize-job", action="append", default=[])
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
            "control_root",
            "tokenizer",
            "benchmark",
            "smoke",
            "shard",
        ],
        required=True,
    )
    path_parser.add_argument("--eval-id")
    path_parser.add_argument("--shard-index", type=int, choices=range(4))
    path_parser.set_defaults(handler=print_path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
