#!/usr/bin/env python3
"""Control, score, aggregate, and audit greedy/sample MATH-500 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from examples.qwen3_8b_32b_full_repro.math500_scorer_v6 import (
    SCORER_VERSION,
    interpret_answer,
    score_response,
    verify_symmetric,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_eval_handoff import (
    checkpoint_manifest as checkpoint_manifests,
)


HANDOFF = Path(__file__).resolve().parent
MANIFEST = HANDOFF / "available_eval_manifest.json"
POLICY = HANDOFF / "eval_policy.json"
EXPERIMENT = Path(
    "checkpoints/qwen_correct_only_openr1_math220k_sft_2k/v1/3568736a3743c319"
)
OUTPUT_NAME = "math500_greedy_sampled_v3"
RESULT_NAME = "math500_greedy_sampled_v3"
TARGETS = (
    "base_qwen2_5_3b",
    "qwen2_5_3b_8k",
    "qwen2_5_3b_16k",
    "base_qwen2_5_7b",
    "qwen2_5_7b_8k",
    "base_qwen3_4b",
    "qwen3_4b_8k",
    "base_qwen3_8b",
    "qwen3_8b_8k",
)
MANIFEST_VARIANTS = {
    "base_qwen2_5_3b": "base_qwen2_5_3b",
    "qwen2_5_3b_8k": "qwen2_5_3b_8k.correct_only",
    "qwen2_5_3b_16k": "qwen2_5_3b_16k.correct_only",
    "base_qwen2_5_7b": "base_qwen2_5_7b",
    "qwen2_5_7b_8k": "qwen2_5_7b_8k.correct_only",
    "base_qwen3_4b": "base_qwen3_4b",
    "qwen3_4b_8k": "qwen3_4b_8k.correct_only",
    "base_qwen3_8b": "base_qwen3_8b",
    "qwen3_8b_8k": "qwen3_8b_8k.correct_only",
}
PAIRS = (
    ("Qwen2.5-3B 8K", "base_qwen2_5_3b", "qwen2_5_3b_8k"),
    ("Qwen2.5-3B 16K", "base_qwen2_5_3b", "qwen2_5_3b_16k"),
    ("Qwen2.5-7B 8K", "base_qwen2_5_7b", "qwen2_5_7b_8k"),
    ("Qwen3-4B 8K", "base_qwen3_4b", "qwen3_4b_8k"),
    ("Qwen3-8B 8K", "base_qwen3_8b", "qwen3_8b_8k"),
)
CANARY_CASES = (
    ("qwen2_5", "base_qwen2_5_3b", "greedy", 0),
    ("qwen3", "base_qwen3_4b", "sampled", 0),
)
RUNTIME_DISTRIBUTIONS = {
    "cffi": "cffi",
    "math_verify": "math-verify",
    "mistral_common": "mistral-common",
    "numpy": "numpy",
    "pycountry": "pycountry",
    "pycparser": "pycparser",
    "pydantic_extra_types": "pydantic-extra-types",
    "soundfile": "soundfile",
    "soxr": "soxr",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "transformers": "transformers",
    "triton": "triton",
    "vllm": "vllm",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root(args: argparse.Namespace) -> Path:
    root = Path(args.repo_root).resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"not a Git worktree: {root}")
    return root


def load_inventory(root: Path, path: str | Path = MANIFEST) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    value = load_json(source)
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or tuple(item.get("id") for item in checkpoints) != TARGETS:
        raise RuntimeError("evaluation inventory target order changed")
    if value.get("contract_hash") != "3568736a3743c319":
        raise RuntimeError("training contract hash changed")
    if value.get("dataset", {}).get("revision") != "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be":
        raise RuntimeError("MATH-500 revision changed")
    if set(value.get("runtime", {})) != set(RUNTIME_DISTRIBUTIONS):
        raise RuntimeError("evaluation runtime pin changed")
    if value.get("sharding") != {
        "array_tasks_per_checkpoint": 16,
        "chunk_size": 32,
        "count": 4,
        "layout": "contiguous",
        "problems_per_shard": 125,
    }:
        raise RuntimeError("evaluation sharding changed")
    policy = load_json(POLICY)
    if policy["stopping"]["accepted"] != [
        {"id": 151643, "token": "<|endoftext|>"},
        {"id": 151645, "token": "<|im_end|>"},
    ]:
        raise RuntimeError("stop-token contract changed")
    if policy["scoring"]["scorer"] != SCORER_VERSION:
        raise RuntimeError("scorer policy changed")
    return value


def item_for(inventory: dict[str, Any], target: str) -> dict[str, Any]:
    matches = [item for item in inventory["checkpoints"] if item["id"] == target]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate target: {target}")
    return matches[0]


def absolute(root: Path, item: dict[str, Any], field: str) -> Path:
    return (root / item[field]).resolve()


def eval_root(root: Path) -> Path:
    return (root / EXPERIMENT / "outputs" / OUTPUT_NAME).resolve()


def control_root(root: Path) -> Path:
    return eval_root(root) / "_control"


def result_root(root: Path) -> Path:
    return (
        root
        / "examples/qwen_correct_only_openr1_math220k_sft_2k/results"
        / RESULT_NAME
    ).resolve()


def tokenizer_root(root: Path, family: str) -> Path:
    return control_root(root) / "tokenizers" / family


def benchmark_paths(root: Path) -> tuple[Path, list[Path], Path]:
    directory = control_root(root) / "eval_benchmarks"
    full = directory / "math500.jsonl"
    shards = [
        directory / f"shard_{index:02d}_{index * 125:03d}_{(index + 1) * 125:03d}.jsonl"
        for index in range(4)
    ]
    return full, shards, directory / "metadata.json"


def pass_spec(task_index: int) -> tuple[str, int, int]:
    if task_index not in range(16):
        raise ValueError(f"array task must be 0..15, found {task_index}")
    pass_index, shard = divmod(task_index, 4)
    return ("greedy", 0, shard) if pass_index == 0 else ("sampled", pass_index - 1, shard)


def pass_key(mode: str, repeat: int) -> str:
    if mode == "greedy" and repeat == 0:
        return "greedy"
    if mode == "sampled" and repeat in range(3):
        return f"sampled/repeat_{repeat:02d}"
    raise ValueError(f"invalid pass: {mode}/{repeat}")


def pass_output(root: Path, target: str, mode: str, repeat: int) -> Path:
    return eval_root(root) / target / pass_key(mode, repeat)


def canary_output(root: Path, family: str, mode: str, repeat: int) -> Path:
    return control_root(root) / "canary" / family / pass_key(mode, repeat)


def validate_runtime_contract(inventory: dict[str, Any]) -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(distribution)
        for name, distribution in RUNTIME_DISTRIBUTIONS.items()
    }
    if versions != inventory["runtime"]:
        raise RuntimeError(
            f"runtime mismatch: actual={versions} expected={inventory['runtime']}"
        )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    probe = (
        "from transformers import AutoTokenizer; "
        "from vllm import LLM, SamplingParams, TokensPrompt; "
        "from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM; "
        "from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM; "
        "print(Qwen2ForCausalLM.__name__, Qwen3ForCausalLM.__name__)"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
    return versions


def validate_runtime(args: argparse.Namespace) -> None:
    inventory = load_inventory(repo_root(args), args.manifest)
    versions = validate_runtime_contract(inventory)
    print(
        f"QCO_RUNTIME_VALIDATED distributions={len(versions)} "
        "architectures=Qwen2ForCausalLM,Qwen3ForCausalLM",
        flush=True,
    )


def normalized_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 500:
        raise RuntimeError(f"MATH-500 source must have 500 rows, found {len(rows)}")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        problem = row.get("problem")
        answer = row.get("correct_answer", row.get("answer"))
        problem_id = row.get("problem_id") or row.get("unique_id") or row.get("id") or f"math500_{index:03d}"
        if not isinstance(problem, str) or not problem.strip() or answer is None:
            raise RuntimeError(f"invalid MATH-500 row {index}")
        result.append(
            {
                "correct_answer": str(answer),
                "level": row.get("level"),
                "problem": problem,
                "problem_id": str(problem_id),
                "subject": row.get("subject"),
            }
        )
    if len({row["problem_id"] for row in result}) != 500:
        raise RuntimeError("MATH-500 contains duplicate problem IDs")
    return result


def prepare_tokenizers(root: Path, inventory: dict[str, Any]) -> None:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        from tokenizers.tokenizers import Tokenizer

    for family, spec in inventory["tokenizer_families"].items():
        base_item = item_for(inventory, spec["base_source"])
        trained_item = item_for(inventory, spec["trained_source"])
        base = absolute(root, base_item, "model")
        trained = absolute(root, trained_item, "model")
        base_vocab = Tokenizer.from_file(str(base / "tokenizer.json")).get_vocab(with_added_tokens=True)
        trained_vocab = Tokenizer.from_file(str(trained / "tokenizer.json")).get_vocab(with_added_tokens=True)
        changed = {token for token, token_id in base_vocab.items() if trained_vocab.get(token) != token_id}
        trained_only = sorted(set(trained_vocab) - set(base_vocab))
        if changed or trained_only != spec["expected_trained_only_tokens"]:
            raise RuntimeError(
                f"{family} tokenizer vocabulary drift: changed={len(changed)} trained_only={trained_only}"
            )
        base_config = load_json(base / "tokenizer_config.json")
        trained_template = (trained / "chat_template.jinja").read_text(encoding="utf-8")
        if base_config.get("chat_template") != trained_template:
            raise RuntimeError(f"{family} base/trained chat templates differ")
        expected_trained_hashes = {
            name: sha256_file(trained / name)
            for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")
        }
        for item in inventory["checkpoints"]:
            if item["tokenizer_family"] != family or item["training_condition"] == "base":
                continue
            model = absolute(root, item, "model")
            actual = {name: sha256_file(model / name) for name in expected_trained_hashes}
            if actual != expected_trained_hashes:
                raise RuntimeError(f"trained tokenizer differs within {family}: {item['id']}")
        destination = tokenizer_root(root, family)
        config = load_json(trained / "tokenizer_config.json")
        original_extra = config.get("extra_special_tokens")
        if not isinstance(original_extra, list):
            raise RuntimeError(f"{family} tokenizer compatibility source changed")
        compatible = dict(config)
        compatible["extra_special_tokens"] = {}
        atomic_text(destination / "tokenizer_config.json", canonical_json(compatible))
        atomic_bytes(destination / "tokenizer.json", (trained / "tokenizer.json").read_bytes())
        atomic_bytes(destination / "chat_template.jinja", trained_template.encode())
        atomic_text(
            destination / "TOKENIZER_PROVENANCE.json",
            canonical_json(
                {
                    "artifact_schema_version": 1,
                    "base_source": spec["base_source"],
                    "compatibility_transform": {"field": "extra_special_tokens", "from": original_extra, "to": {}},
                    "family": family,
                    "shared_vocabulary_ids_equal": True,
                    "trained_only_tokens": trained_only,
                    "trained_source": spec["trained_source"],
                    "files": {
                        name: sha256_file(destination / name)
                        for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")
                    },
                }
            ),
        )


def prepare(args: argparse.Namespace) -> None:
    root = repo_root(args)
    inventory = load_inventory(root, args.manifest)
    source = Path(args.source).resolve()
    rows = normalized_source_rows(read_jsonl(source))
    canonical = jsonl_text(rows)
    expected = inventory["dataset"]["axolotl_canonical_jsonl_sha256"]
    if sha256_bytes(canonical.encode()) != expected:
        raise RuntimeError("canonical MATH-500 source hash changed")
    indexed = [dict(row, eval_index=index) for index, row in enumerate(rows)]
    full, shards, metadata_path = benchmark_paths(root)
    atomic_text(full, jsonl_text(indexed))
    shard_records = []
    for index, path in enumerate(shards):
        start, end = index * 125, (index + 1) * 125
        atomic_text(path, jsonl_text(indexed[start:end]))
        shard_records.append({"index": index, "path": str(path), "sha256": sha256_file(path), "start": start, "end": end})
    atomic_text(full.parent / "smoke.jsonl", jsonl_text(indexed[:1]))
    atomic_text(
        metadata_path,
        canonical_json(
            {
                "artifact_schema_version": 1,
                "canonical_source_sha256": expected,
                "indexed_full_sha256": sha256_file(full),
                "rows": 500,
                "shards": shard_records,
                "source": str(source),
                "source_sha256": sha256_file(source),
            }
        ),
    )
    prepare_tokenizers(root, inventory)
    print(f"MATH500_EVAL_PREPARED rows=500 root={full.parent}", flush=True)


def quick_checkpoint(checkpoint: Path, manifest_path: Path) -> str:
    manifest = load_json(manifest_path)
    checkpoint_manifests.validate_manifest(manifest)
    expected = {entry["name"]: int(entry["bytes"]) for entry in manifest["files"]}
    actual = {
        path.relative_to(checkpoint).as_posix(): path.stat().st_size
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"checkpoint size inventory mismatch: {checkpoint}")
    return str(manifest["checkpoint_manifest_sha256"])


def verify_checkpoints(args: argparse.Namespace) -> None:
    root = repo_root(args)
    inventory = load_inventory(root, args.manifest)
    identities = {}
    for item in inventory["checkpoints"]:
        model = absolute(root, item, "model")
        manifest_path = absolute(root, item, "manifest")
        identity = (
            checkpoint_manifests.verify(manifest_path, model)
            if args.full
            else quick_checkpoint(model, manifest_path)
        )
        manifest = load_json(manifest_path)
        is_base = item["training_condition"] == "base"
        if manifest.get("variant") != MANIFEST_VARIANTS[item["id"]]:
            raise RuntimeError(f"manifest variant mismatch: {item['id']}")
        if is_base and (manifest.get("family") != "shared" or manifest.get("final_iteration") is not None):
            raise RuntimeError(f"invalid base checkpoint: {item['id']}")
        if not is_base and (manifest.get("family") != "2k" or manifest.get("final_iteration") != 124):
            raise RuntimeError(f"invalid trained checkpoint: {item['id']}")
        identities[item["id"]] = identity
        print(f"CHECKPOINT_VERIFIED target={item['id']} full={int(args.full)} sha256={identity}", flush=True)
    if args.status:
        atomic_text(Path(args.status), canonical_json({"artifact_schema_version": 1, "checked_at": now_iso(), "full_hash_verification": bool(args.full), "identities": identities}))


def validate_gold(args: argparse.Namespace) -> None:
    root = repo_root(args)
    full = benchmark_paths(root)[0]
    failures = []
    for row in read_jsonl(full):
        gold = str(row["correct_answer"])
        if not interpret_answer(gold) or not verify_symmetric((gold,), gold)["equivalent"]:
            failures.append({"problem_id": row["problem_id"], "gold": gold})
    if failures or len(read_jsonl(full)) != 500:
        raise RuntimeError(f"gold self-verification failed: {failures[:10]}")
    atomic_text(control_root(root) / "GOLD_COVERAGE.json", canonical_json({"rows": 500, "scorer_version": SCORER_VERSION, "status": "complete"}))


def records_for_pass(root: Path, target: str, mode: str, repeat: int) -> list[dict[str, Any]]:
    _, shards, _ = benchmark_paths(root)
    records: list[dict[str, Any]] = []
    item = item_for(load_inventory(root), target)
    checkpoint = load_json(absolute(root, item, "manifest"))["checkpoint_manifest_sha256"]
    for shard_index, eval_file in enumerate(shards):
        expected = read_jsonl(eval_file)
        expected_indices = [int(row["eval_index"]) for row in expected]
        output = pass_output(root, target, mode, repeat) / "shards" / f"shard_{shard_index:02d}"
        rows = read_jsonl(output / "records.jsonl")
        by_index = {int(row["eval_index"]): row for row in rows}
        if len(rows) != 125 or set(by_index) != set(expected_indices):
            raise RuntimeError(f"incomplete generation shard: {output}")
        generation_policy = load_json(output / "generation_policy.json")
        fingerprint = sha256_bytes(json.dumps(generation_policy, sort_keys=True, separators=(",", ":")).encode())
        for row in expected:
            record = by_index[int(row["eval_index"])]
            if (
                record.get("target") != target
                or record.get("mode") != mode
                or int(record.get("repeat", -1)) != repeat
                or record.get("checkpoint_manifest_sha256") != checkpoint
                or record.get("policy_sha256") != fingerprint
                or int(record.get("rendered_prompt_tokens", 0)) + int(record.get("effective_max_response_tokens", 0)) != 32768
                or sha256_bytes(str(record.get("response", "")).encode()) != record.get("response_sha256")
            ):
                raise RuntimeError(f"generation provenance mismatch: {target}/{mode}/{repeat}/{row['eval_index']}")
            records.append(record)
    if [int(row["eval_index"]) for row in records] != list(range(500)):
        raise RuntimeError(f"merged generation order changed: {target}/{mode}/{repeat}")
    return records


def score_pass(root: Path, target: str, mode: str, repeat: int) -> dict[str, Any]:
    full_rows = read_jsonl(benchmark_paths(root)[0])
    records = records_for_pass(root, target, mode, repeat)
    scored = []
    for benchmark, record in zip(full_rows, records, strict=True):
        trace = score_response(
            str(record["response"]),
            str(benchmark["correct_answer"]),
            finish_reason=record.get("finish_reason"),
            stop_token_id=(
                int(record["stop_reason"])
                if isinstance(record.get("stop_reason"), int)
                else None
            ),
            cap_hit=bool(record["cap_hit"]),
            problem_id=str(benchmark["problem_id"]),
        )
        scored.append({"generation": record, "gold_answer": str(benchmark["correct_answer"]), "problem_id": str(benchmark["problem_id"]), "scorer_trace": trace})
    correct = sum(bool(row["scorer_trace"]["is_correct"]) for row in scored)
    valid = sum(row["scorer_trace"]["official_extracted_source_span"] is not None for row in scored)
    caps = sum(bool(row["generation"]["cap_hit"]) for row in scored)
    cap_correct = sum(bool(row["scorer_trace"]["is_correct"]) for row in scored if row["generation"]["cap_hit"])
    parse_failures = sum(row["scorer_trace"]["official_decision_reason"] == "parse_failure" for row in scored)
    output = pass_output(root, target, mode, repeat) / "merged"
    atomic_text(output / "scored.jsonl", jsonl_text(scored))
    summary = {
        "accuracy": correct / 500,
        "cap_hit_accuracy": cap_correct / caps if caps else 0.0,
        "cap_hit_correct": cap_correct,
        "cap_hit_rate": caps / 500,
        "cap_hits": caps,
        "correct": correct,
        "finish_reasons": dict(sorted(Counter(str(row["generation"].get("finish_reason")) for row in scored).items())),
        "generated_length_mean": statistics.fmean(int(row["generation"]["generated_token_count"]) for row in scored),
        "mode": mode,
        "parse_failure_count": parse_failures,
        "repeat": repeat,
        "scorer_version": SCORER_VERSION,
        "target": target,
        "total": 500,
        "valid_box_count": valid,
        "valid_box_rate": valid / 500,
    }
    atomic_text(output / "summary.json", canonical_json(summary))
    atomic_text(output / "summary.sha256", sha256_file(output / "summary.json") + "\n")
    return summary


def mean_sd(values: list[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise ValueError(f"expected three sampled values, found {len(values)}")
    return {"mean": statistics.fmean(values), "sample_std": statistics.stdev(values), "values": values}


def merge(args: argparse.Namespace) -> None:
    root = repo_root(args)
    inventory = load_inventory(root, args.manifest)
    item = item_for(inventory, args.target)
    greedy = score_pass(root, args.target, "greedy", 0)
    sampled = [score_pass(root, args.target, "sampled", repeat) for repeat in range(3)]
    aggregate_fields = ("accuracy", "valid_box_rate", "cap_hit_rate", "cap_hit_accuracy", "generated_length_mean")
    result = {
        "artifact_schema_version": 1,
        "benchmark": "HuggingFaceH4/MATH-500",
        "checkpoint_manifest_sha256": load_json(absolute(root, item, "manifest"))["checkpoint_manifest_sha256"],
        "dataset_revision": inventory["dataset"]["revision"],
        "greedy": greedy,
        "policy_sha256": sha256_file(POLICY),
        "sampled": {
            "aggregate": {field: mean_sd([float(row[field]) for row in sampled]) for field in aggregate_fields},
            "metric": "mean and sample standard deviation across three complete pass@1 repeats",
            "repeats": sampled,
        },
        "scorer_version": SCORER_VERSION,
        "target": args.target,
        "tokenizer_family": item["tokenizer_family"],
        "training_condition": item["training_condition"],
    }
    output = eval_root(root) / args.target / "summary.json"
    atomic_text(output, canonical_json(result))
    destination = result_root(root) / args.target
    destination.mkdir(parents=True, exist_ok=True)
    sampled_accuracy_text = ", ".join(
        f"{row['accuracy']:.2%}" for row in sampled
    )
    files = {
        "RESULTS.json": canonical_json(result),
        "RESULTS.md": (
            f"# {args.target} MATH-500\n\n"
            f"- Greedy: {greedy['correct']}/500 ({greedy['accuracy']:.2%})\n"
            f"- Sampled: {result['sampled']['aggregate']['accuracy']['mean']:.2%} ± "
            f"{result['sampled']['aggregate']['accuracy']['sample_std']:.2%} SD\n"
            f"- Sample repeats: {sampled_accuracy_text}\n"
        ),
        "EVAL_POLICY.json": canonical_json(load_json(POLICY)),
        "CHECKPOINT.json": canonical_json(load_json(absolute(root, item, "manifest"))),
        "PROVENANCE.json": canonical_json({"artifact_schema_version": 1, "generated_results": str(eval_root(root) / args.target), "policy_sha256": sha256_file(POLICY), "scorer_commit": args.scorer_commit, "tokenizer_provenance_sha256": sha256_file(tokenizer_root(root, item["tokenizer_family"]) / "TOKENIZER_PROVENANCE.json")}),
    }
    for name, content in files.items():
        path = destination / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing incompatible compact result: {path}")
        atomic_text(path, content)
    print(f"MATH500_TARGET_FINALIZED target={args.target} greedy={greedy['correct']}/500 sampled={result['sampled']['aggregate']['accuracy']['mean']:.4f}", flush=True)


def audit(args: argparse.Namespace) -> None:
    root = repo_root(args)
    results = {target: load_json(result_root(root) / target / "RESULTS.json") for target in TARGETS}
    rows = []
    for target in TARGETS:
        result = results[target]
        if result.get("target") != target or result.get("scorer_version") != SCORER_VERSION:
            raise RuntimeError(f"invalid final result: {target}")
        rows.append({"target": target, "greedy_correct": result["greedy"]["correct"], "greedy_accuracy": result["greedy"]["accuracy"], "sampled_accuracy_mean": result["sampled"]["aggregate"]["accuracy"]["mean"], "sampled_accuracy_sample_std": result["sampled"]["aggregate"]["accuracy"]["sample_std"]})
    comparisons = []
    for label, base_name, trained_name in PAIRS:
        base, trained = results[base_name], results[trained_name]
        base_values = base["sampled"]["aggregate"]["accuracy"]["values"]
        trained_values = trained["sampled"]["aggregate"]["accuracy"]["values"]
        deltas = [float(t) - float(b) for b, t in zip(base_values, trained_values, strict=True)]
        comparisons.append({"comparison": label, "base": base_name, "trained": trained_name, "greedy_delta": trained["greedy"]["accuracy"] - base["greedy"]["accuracy"], "sampled_delta": mean_sd(deltas)})
    value = {"artifact_schema_version": 1, "audited_at": now_iso(), "comparisons": comparisons, "results": rows, "scorer_version": SCORER_VERSION, "status": "complete"}
    atomic_text(control_root(root) / "FINAL_AUDIT.json", canonical_json(value))
    atomic_text(result_root(root) / "RESULTS.json", canonical_json(value))
    lines = [
        f"# Correct-only Qwen MATH-500 ({SCORER_VERSION})",
        "",
        "| Target | Greedy | Sampled mean ± SD | Repeat accuracies |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGETS:
        result = results[target]
        agg = result["sampled"]["aggregate"]["accuracy"]
        lines.append(f"| {target} | {result['greedy']['correct']}/500 ({result['greedy']['accuracy']:.2%}) | {agg['mean']:.2%} ± {agg['sample_std']:.2%} | {', '.join(f'{value:.2%}' for value in agg['values'])} |")
    lines.extend(["", "| Comparison | Greedy Δ | Sampled paired Δ mean ± SD |", "|---|---:|---:|"])
    for row in comparisons:
        lines.append(f"| {row['comparison']} | {row['greedy_delta']:+.2%} | {row['sampled_delta']['mean']:+.2%} ± {row['sampled_delta']['sample_std']:.2%} |")
    atomic_text(result_root(root) / "RESULTS.md", "\n".join(lines) + "\n")
    print(f"MATH500_ALL_RESULTS_COMPLETE targets=9 comparisons=5 audit={control_root(root) / 'FINAL_AUDIT.json'}", flush=True)


def mark_preflight(args: argparse.Namespace) -> None:
    root = repo_root(args)
    inventory = load_inventory(root, args.manifest)
    versions = validate_runtime_contract(inventory)
    verification = load_json(Path(args.verification))
    if verification.get("full_hash_verification") is not True or set(verification.get("identities", {})) != set(TARGETS):
        raise RuntimeError("full checkpoint verification is incomplete")
    tokenizer_hashes = {family: sha256_file(tokenizer_root(root, family) / "TOKENIZER_PROVENANCE.json") for family in inventory["tokenizer_families"]}
    atomic_text(control_root(root) / "PREFLIGHT_READY.json", canonical_json({"artifact_schema_version": 1, "job_id": args.job_id, "ready_at": now_iso(), "runtime": versions, "status": "ready", "tokenizers": tokenizer_hashes, "verification_sha256": sha256_file(Path(args.verification))}))


def check_preflight(args: argparse.Namespace) -> None:
    ready = load_json(control_root(repo_root(args)) / "PREFLIGHT_READY.json")
    if ready.get("status") != "ready":
        raise RuntimeError("preflight is not ready")


def mark_canary(args: argparse.Namespace) -> None:
    root = repo_root(args)
    completed = []
    for family, target, mode, repeat in CANARY_CASES:
        rows = read_jsonl(
            canary_output(root, family, mode, repeat) / "records.jsonl"
        )
        if (
            len(rows) != 1
            or rows[0].get("target") != target
            or rows[0].get("mode") != mode
            or int(rows[0].get("repeat", -1)) != repeat
        ):
            raise RuntimeError(f"invalid {family}/{mode} canary")
        completed.append(
            {"family": family, "mode": mode, "repeat": repeat, "target": target}
        )
    atomic_text(
        control_root(root) / "CANARY_READY.json",
        canonical_json(
            {
                "artifact_schema_version": 1,
                "cases": completed,
                "job_id": args.job_id,
                "ready_at": now_iso(),
                "status": "ready",
            }
        ),
    )


def check_canary(args: argparse.Namespace) -> None:
    check_preflight(args)
    ready = load_json(control_root(repo_root(args)) / "CANARY_READY.json")
    if ready.get("status") != "ready":
        raise RuntimeError("canary is not ready")


def record_submission(args: argparse.Namespace) -> None:
    root = repo_root(args)
    output = control_root(root) / "submission.json"
    if output.exists():
        raise FileExistsError(f"refusing duplicate submission: {output}")
    def assignments(values: list[str]) -> dict[str, str]:
        result = {}
        for value in values:
            name, separator, job = value.partition("=")
            if separator != "=" or name in result or name not in TARGETS or not job.isdigit():
                raise ValueError(f"invalid job assignment: {value}")
            result[name] = job
        return result
    arrays, finalizers = assignments(args.array_job), assignments(args.finalizer_job)
    if set(arrays) != set(TARGETS) or set(finalizers) != set(TARGETS):
        raise RuntimeError("submission does not cover all targets")
    value = {
        "artifact_schema_version": 1,
        "attempt": 3,
        "array_jobs": arrays,
        "audit_job": args.audit_job,
        "canary_job": args.canary_job,
        "eval_task_count": 144,
        "finalizer_jobs": finalizers,
        "git_commit": args.git_commit,
        "output_name": OUTPUT_NAME,
        "preflight_job": args.preflight_job,
        "slurm": {
            "account": "raivn-ckpt",
            "array": "0-15",
            "canary_requeue_signal_lead_seconds": 300,
            "canary_walltime": "00:30:00",
            "excluded_h200_nodes": ["g3130"],
            "generation_walltime": "02:00:00",
            "partition": "ckpt-all",
            "qos": "ckpt",
        },
        "submitted_at": now_iso(),
    }
    atomic_text(output, canonical_json(value))
    atomic_text(HANDOFF / "submission_metadata.json", canonical_json(value))


def print_path(args: argparse.Namespace) -> None:
    root = repo_root(args)
    inventory = load_inventory(root, args.manifest)
    if args.field == "control":
        path = control_root(root)
    elif args.field == "smoke":
        path = benchmark_paths(root)[0].parent / "smoke.jsonl"
    elif args.field == "shard":
        path = benchmark_paths(root)[1][args.shard]
    elif args.field == "tokenizer":
        path = tokenizer_root(root, item_for(inventory, args.target)["tokenizer_family"])
    elif args.field in {"model", "manifest"}:
        path = absolute(root, item_for(inventory, args.target), args.field)
    elif args.field == "checkpoint-identity":
        print(load_json(absolute(root, item_for(inventory, args.target), "manifest"))["checkpoint_manifest_sha256"])
        return
    elif args.field == "canary-output":
        item = item_for(inventory, args.target)
        path = canary_output(
            root, item["tokenizer_family"], args.mode, args.repeat
        )
    elif args.field == "pass-output":
        path = pass_output(root, args.target, args.mode, args.repeat)
    else:
        raise RuntimeError(f"unknown path field: {args.field}")
    print(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", required=True)
    value.add_argument("--manifest", default=str(MANIFEST))
    commands = value.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare"); prepare_cmd.add_argument("--source", required=True)
    verify_cmd = commands.add_parser("verify-checkpoints"); verify_cmd.add_argument("--full", action="store_true"); verify_cmd.add_argument("--status")
    commands.add_parser("validate-runtime")
    commands.add_parser("validate-gold")
    merge_cmd = commands.add_parser("merge"); merge_cmd.add_argument("--target", required=True, choices=TARGETS); merge_cmd.add_argument("--scorer-commit", required=True)
    commands.add_parser("audit")
    preflight_cmd = commands.add_parser("mark-preflight"); preflight_cmd.add_argument("--job-id", required=True); preflight_cmd.add_argument("--verification", required=True)
    commands.add_parser("check-preflight")
    canary_cmd = commands.add_parser("mark-canary"); canary_cmd.add_argument("--job-id", required=True)
    commands.add_parser("check-canary")
    record_cmd = commands.add_parser("record-submission"); record_cmd.add_argument("--preflight-job", required=True); record_cmd.add_argument("--canary-job", required=True); record_cmd.add_argument("--array-job", action="append", default=[]); record_cmd.add_argument("--finalizer-job", action="append", default=[]); record_cmd.add_argument("--audit-job", required=True); record_cmd.add_argument("--git-commit", required=True)
    path_cmd = commands.add_parser("path"); path_cmd.add_argument("--field", required=True, choices=["control", "smoke", "shard", "tokenizer", "model", "manifest", "checkpoint-identity", "canary-output", "pass-output"]); path_cmd.add_argument("--target", choices=TARGETS); path_cmd.add_argument("--mode", choices=["greedy", "sampled"]); path_cmd.add_argument("--repeat", type=int, default=0); path_cmd.add_argument("--shard", type=int, choices=range(4), default=0)
    return value


def main() -> None:
    args = parser().parse_args()
    handlers = {"prepare": prepare, "verify-checkpoints": verify_checkpoints, "validate-runtime": validate_runtime, "validate-gold": validate_gold, "merge": merge, "audit": audit, "mark-preflight": mark_preflight, "check-preflight": check_preflight, "mark-canary": mark_canary, "check-canary": check_canary, "record-submission": record_submission, "path": print_path}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
