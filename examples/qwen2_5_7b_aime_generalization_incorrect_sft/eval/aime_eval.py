#!/usr/bin/env python3
"""Prepare, verify, score, and report the AIME generalization evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.aime_scorer import (
    SCORER_VERSION,
    parse_aime_integer,
    score_response,
)
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.generate_aime import (
    DECODING,
    STOP_TOKENS,
    TARGET_CONTEXT,
    generation_policy,
    sample_seed,
)
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.tokenizer_compat import (
    QWEN_SPECIAL_TOKENS,
    expected_overlay_path,
    overlay_spec,
    prepare_overlay,
    validate_overlay,
)
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.hf_checkpoint_gate import (
    validate_checkpoint,
)

PACKAGE = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
EVAL_CONFIG = EVAL_DIR / "eval_config.yaml"
EVAL_NAME = "aime_generalization_sampled_v1"
SPLITS = ("held_in", "held_out_problem")
TARGETS = (
    "base_qwen2_5_7b",
    "aime_correct_3000",
    "continue_wrong_unmasked",
    "continue_random_mask_10",
    "continue_random_mask_20",
    "continue_random_mask_30",
    "continue_random_mask_40",
    "continue_random_mask_50",
    "continue_random_mask_60",
    "continue_random_mask_70",
    "continue_random_mask_80",
    "continue_random_mask_90",
    "continue_correct_only_1500",
)
BASE_TARGET = "base_qwen2_5_7b"
PRIORITY_TARGETS = (
    "aime_correct_3000",
    "continue_wrong_unmasked",
    "continue_correct_only_1500",
    "continue_random_mask_50",
)
INTERIM_TARGETS = (BASE_TARGET, *PRIORITY_TARGETS)
DEFERRED_TARGETS = (
    "continue_random_mask_10",
    "continue_random_mask_20",
    "continue_random_mask_30",
    "continue_random_mask_40",
    "continue_random_mask_60",
    "continue_random_mask_70",
    "continue_random_mask_80",
    "continue_random_mask_90",
)
REFERENCE_TARGETS = (BASE_TARGET, "aime_correct_3000")
PROBLEMS_PER_SPLIT = 400
REPEATS = 3
SHARDS = 4
PROBLEMS_PER_SHARD = 100
PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
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


def experiment_root(args: argparse.Namespace) -> Path:
    return Path(args.experiment_root).resolve()


def inventory_path(args: argparse.Namespace) -> Path:
    return experiment_root(args) / "handoff/comparison_sources.json"


def load_inventory(args: argparse.Namespace) -> dict[str, Any]:
    value = load_json(inventory_path(args))
    targets = value.get("targets")
    if not isinstance(targets, list):
        raise RuntimeError("comparison inventory has no target list")
    actual = tuple(str(item.get("id")) for item in targets)
    if actual != TARGETS:
        raise RuntimeError(f"comparison target order changed: {actual}")
    if value.get("comparison_checkpoint_count") != len(TARGETS):
        raise RuntimeError("comparison checkpoint count changed")
    for item in targets:
        identity = item.get("checkpoint_identity")
        if not isinstance(identity, str) or len(identity) != 64:
            raise RuntimeError(f"invalid checkpoint identity: {item.get('id')}")
    return value


def item_for(inventory: dict[str, Any], target: str) -> dict[str, Any]:
    matches = [item for item in inventory["targets"] if item["id"] == target]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate target: {target}")
    return matches[0]


def remote_artifact_path(
    root: Path,
    current_experiment: Path,
    item: dict[str, Any],
    relative_field: str,
    source_field: str,
) -> Path:
    relative = item.get(relative_field)
    remote_experiment = item.get("remote_experiment_relative")
    if isinstance(relative, str) and relative:
        base = (
            (root / remote_experiment).resolve()
            if isinstance(remote_experiment, str) and remote_experiment
            else current_experiment
        )
        return (base / relative).resolve()
    source = item.get(source_field)
    if isinstance(source, str) and source:
        return Path(source).resolve()
    raise RuntimeError(f"target {item.get('id')} lacks {relative_field}")


def model_path(args: argparse.Namespace, item: dict[str, Any]) -> Path:
    return remote_artifact_path(
        repo_root(args),
        experiment_root(args),
        item,
        "remote_checkpoint_relative",
        "source_checkpoint",
    )


def manifest_path(args: argparse.Namespace, item: dict[str, Any]) -> Path:
    return remote_artifact_path(
        repo_root(args),
        experiment_root(args),
        item,
        "remote_manifest_relative",
        "source_manifest",
    )


def eval_root(args: argparse.Namespace) -> Path:
    return experiment_root(args) / "outputs/eval" / EVAL_NAME


def control_root(args: argparse.Namespace) -> Path:
    return eval_root(args) / "_control"


def tokenizer_overlay_root(args: argparse.Namespace) -> Path:
    return control_root(args) / "tokenizer_overlays"


def tokenizer_path(args: argparse.Namespace, target: str, item: dict[str, Any]) -> Path:
    model = model_path(args, item)
    identity = str(item["checkpoint_identity"])
    path = expected_overlay_path(tokenizer_overlay_root(args), target, model, identity)
    _, metadata = overlay_spec(model, identity)
    validate_overlay(path, metadata)
    return path


def result_root(args: argparse.Namespace) -> Path:
    return experiment_root(args) / "results" / EVAL_NAME


def benchmark_full(args: argparse.Namespace, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"invalid split: {split}")
    return control_root(args) / "eval_benchmarks" / split / "full.jsonl"


def benchmark_shard(args: argparse.Namespace, split: str, shard: int) -> Path:
    if shard not in range(SHARDS):
        raise ValueError(f"invalid shard: {shard}")
    start = shard * PROBLEMS_PER_SHARD
    end = start + PROBLEMS_PER_SHARD
    return control_root(args) / "eval_benchmarks" / split / f"shard_{shard:02d}_{start:03d}_{end:03d}.jsonl"


def pass_output(args: argparse.Namespace, target: str, split: str, repeat: int) -> Path:
    if target not in TARGETS or split not in SPLITS or repeat not in range(REPEATS):
        raise ValueError(f"invalid pass coordinate: {target}/{split}/{repeat}")
    return eval_root(args) / target / split / f"repeat_{repeat:02d}"


def normalize_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != PROBLEMS_PER_SPLIT:
        raise RuntimeError(f"{split} must contain exactly {PROBLEMS_PER_SPLIT} rows, found {len(rows)}")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        problem_id = row.get("problem_id")
        prompt = row.get("prompt")
        question = row.get("question")
        answer = row.get("answer")
        year = row.get("year")
        part = row.get("part")
        problem_number = row.get("problem_number")
        source_split = row.get("split")
        if not isinstance(problem_id, str) or not problem_id:
            raise RuntimeError(f"invalid problem_id at {split}/{index}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"invalid prompt at {split}/{index}")
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError(f"invalid question at {split}/{index}")
        if prompt != question + PROMPT_SUFFIX:
            raise RuntimeError(f"noncanonical prompt at {split}/{index}")
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer <= 999:
            raise RuntimeError(f"invalid answer at {split}/{index}")
        if isinstance(year, bool) or not isinstance(year, int) or year == 2024:
            raise RuntimeError(f"invalid/excluded year at {split}/{index}: {year}")
        if part is not None and part not in {"I", "II"}:
            raise RuntimeError(f"invalid AIME part at {split}/{index}: {part}")
        if isinstance(problem_number, bool) or not isinstance(problem_number, int):
            raise RuntimeError(f"invalid problem_number at {split}/{index}")
        if source_split != split:
            raise RuntimeError(f"invalid source split at {split}/{index}: {source_split}")
        result.append(
            {
                "answer": answer,
                "eval_index": index,
                "part": part,
                "problem_id": problem_id,
                "problem_number": problem_number,
                "prompt": prompt,
                "question": question,
                "source_split": source_split,
                "year": year,
            }
        )
    if len({row["problem_id"] for row in result}) != PROBLEMS_PER_SPLIT:
        raise RuntimeError(f"duplicate problem IDs in {split}")
    return result


def validate_eval_config(held_in: Path, held_out: Path) -> None:
    import yaml

    config = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("eval-config is not a mapping")
    eval_value = config.get("eval")
    if not isinstance(eval_value, dict):
        raise RuntimeError("eval-config is missing eval mapping")
    defaults = eval_value.get("defaults")
    datasets = eval_value.get("datasets")
    if defaults != {
        "n_samples_per_eval_prompt": 1,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": -1,
        "max_response_len": 32768,
        "input_key": "prompt",
        "label_key": "answer",
        "apply_chat_template": True,
    }:
        raise RuntimeError("eval-config defaults changed")
    if not isinstance(datasets, list) or [item.get("name") for item in datasets] != list(SPLITS):
        raise RuntimeError("eval-config dataset order changed")
    expected_paths = (
        (held_in, "${oc.env:HELD_IN_EVAL_JSONL}"),
        (held_out, "${oc.env:HELD_OUT_EVAL_JSONL}"),
    )
    for item, (source_path, path_reference), split in zip(datasets, expected_paths, SPLITS, strict=True):
        if (
            item.get("path") != path_reference
            or item.get("input_key") != "prompt"
            or item.get("label_key") != "answer"
            or item.get("metadata_overrides", {}).get("distribution") != split
            or not source_path.is_file()
        ):
            raise RuntimeError(f"eval-config contract changed: {split}")


def prepare(args: argparse.Namespace) -> None:
    held_in_source = Path(args.held_in).resolve()
    held_out_source = Path(args.held_out).resolve()
    validate_eval_config(held_in_source, held_out_source)
    by_split = {
        "held_in": normalize_rows(held_in_source, "held_in"),
        "held_out_problem": normalize_rows(held_out_source, "held_out_problem"),
    }
    held_ids = {row["problem_id"] for row in by_split["held_in"]}
    held_out_ids = {row["problem_id"] for row in by_split["held_out_problem"]}
    if held_ids & held_out_ids:
        raise RuntimeError("held-in and held-out problem IDs overlap")
    held_contests = {(row["year"], row["part"]) for row in by_split["held_in"]}
    held_out_contests = {(row["year"], row["part"]) for row in by_split["held_out_problem"]}
    contest_overlap = sorted(
        held_contests & held_out_contests,
        key=lambda value: (value[0], value[1] or ""),
    )

    metadata: dict[str, Any] = {
        "artifact_schema_version": 1,
        "created_at": now_iso(),
        "eval_config_sha256": sha256_file(EVAL_CONFIG),
        "problem_id_overlap": 0,
        "contest_overlap_count": len(contest_overlap),
        "contest_overlap": [{"year": year, "part": part} for year, part in contest_overlap],
        "splits": {},
    }
    for split, rows in by_split.items():
        full = benchmark_full(args, split)
        atomic_text(full, jsonl_text(rows))
        atomic_text(full.parent / "smoke.jsonl", jsonl_text(rows[:1]))
        shards = []
        for shard in range(SHARDS):
            start = shard * PROBLEMS_PER_SHARD
            end = start + PROBLEMS_PER_SHARD
            path = benchmark_shard(args, split, shard)
            atomic_text(path, jsonl_text(rows[start:end]))
            shards.append(
                {
                    "end": end,
                    "index": shard,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "start": start,
                }
            )
        source = held_in_source if split == "held_in" else held_out_source
        metadata["splits"][split] = {
            "contest_count": len({(row["year"], row["part"]) for row in rows}),
            "indexed_sha256": sha256_file(full),
            "rows": len(rows),
            "shards": shards,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
        }
    atomic_text(control_root(args) / "eval_benchmarks/metadata.json", canonical_json(metadata))
    print(
        "AIME_EVAL_PREPARED held_in=400 held_out_problem=400 " f"root={control_root(args) / 'eval_benchmarks'}",
        flush=True,
    )


def quick_checkpoint(checkpoint: Path, manifest: Path) -> str:
    value = load_json(manifest)
    expected = {str(row["name"]): int(row["bytes"]) for row in value["files"]}
    actual = {
        path.relative_to(checkpoint).as_posix(): path.stat().st_size
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"checkpoint size inventory mismatch: {checkpoint}")
    identity = str(value.get("checkpoint_manifest_sha256"))
    if len(identity) != 64:
        raise RuntimeError(f"invalid checkpoint identity: {manifest}")
    return identity


def tokenizer_file_contract(model: Path) -> dict[str, Any]:
    tokenizer_json = model / "tokenizer.json"
    tokenizer_config_path = model / "tokenizer_config.json"
    if not tokenizer_json.is_file() or not tokenizer_config_path.is_file():
        raise FileNotFoundError(f"missing required tokenizer files: {model}")
    tokenizer_config = load_json(tokenizer_config_path)
    embedded = tokenizer_config.get("chat_template")
    embedded_template = embedded if isinstance(embedded, str) and embedded else None
    external_path = model / "chat_template.jinja"
    external_template = external_path.read_text(encoding="utf-8") if external_path.is_file() else None
    if embedded_template is None and external_template is None:
        raise RuntimeError(f"checkpoint has no Qwen chat template: {model}")
    if embedded_template is not None and external_template is not None and embedded_template != external_template:
        raise RuntimeError(f"embedded and external chat templates differ: {model}")
    effective = external_template or embedded_template
    assert effective is not None
    return {
        "chat_template_embedded_sha256": (sha256_bytes(embedded_template.encode()) if embedded_template else None),
        "chat_template_external_sha256": (sha256_file(external_path) if external_template is not None else None),
        "chat_template_sha256": sha256_bytes(effective.encode()),
        "eos_token": tokenizer_config.get("eos_token"),
        "pad_token": tokenizer_config.get("pad_token"),
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
    }


def verify_checkpoints(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    inventory = load_inventory(args)
    manifest_tool = (
        repo_root(args) / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
    )
    identities: dict[str, str] = {}
    tokenizer_files: dict[str, dict[str, Any]] = {}
    tokenizer_semantics: dict[str, Any] | None = None
    probe_messages = [
        {
            "role": "user",
            "content": "Tokenizer contract probe.\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
        }
    ]
    for target in TARGETS:
        item = item_for(inventory, target)
        model = model_path(args, item)
        manifest = manifest_path(args, item)
        validate_checkpoint(model)
        if args.full:
            subprocess.run(
                [
                    sys.executable,
                    str(manifest_tool),
                    "verify",
                    "--manifest",
                    str(manifest),
                    "--checkpoint",
                    str(model),
                ],
                check=True,
            )
            identity = str(load_json(manifest)["checkpoint_manifest_sha256"])
        else:
            identity = quick_checkpoint(model, manifest)
        if identity != item["checkpoint_identity"]:
            raise RuntimeError(f"checkpoint identity mismatch: {target}")
        file_contract = tokenizer_file_contract(model)
        overlay, overlay_metadata = prepare_overlay(
            tokenizer_overlay_root(args),
            target,
            model,
            identity,
        )
        file_contract["compatibility_overlay"] = overlay_metadata
        tokenizer_files[target] = file_contract
        tokenizer = AutoTokenizer.from_pretrained(overlay, trust_remote_code=True, use_fast=True)
        rendered = tokenizer.apply_chat_template(probe_messages, tokenize=False, add_generation_prompt=True)
        probe_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
        current_semantics = {
            "chat_template_sha256": file_contract["chat_template_sha256"],
            "eos_token_id": int(tokenizer.eos_token_id),
            "im_end_token_id": int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
            "rendered_probe_sha256": sha256_bytes(rendered.encode()),
            "rendered_probe_token_ids_sha256": sha256_bytes(json.dumps(probe_ids, separators=(",", ":")).encode()),
            "special_token_ids": {
                token: int(tokenizer.convert_tokens_to_ids(token)) for token in QWEN_SPECIAL_TOKENS
            },
            "tokenizer_length": len(tokenizer),
        }
        if current_semantics["eos_token_id"] != 151643 or current_semantics["im_end_token_id"] != 151645:
            raise RuntimeError(f"Qwen stop-token IDs changed: {target}")
        if tokenizer_semantics is None:
            tokenizer_semantics = current_semantics
        elif current_semantics != tokenizer_semantics:
            raise RuntimeError(f"tokenizer semantics differ across checkpoints: {target}")
        identities[target] = identity
        print(
            f"AIME_CHECKPOINT_VERIFIED target={target} full={int(args.full)} " f"sha256={identity}",
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
                    "tokenizer_files": tokenizer_files,
                    "tokenizer_semantics": tokenizer_semantics,
                }
            ),
        )


def records_for_pass(args: argparse.Namespace, target: str, split: str, repeat: int) -> list[dict[str, Any]]:
    inventory = load_inventory(args)
    item = item_for(inventory, target)
    checkpoint = item["checkpoint_identity"]
    overlay_metadata = load_json(tokenizer_path(args, target, item) / "overlay_metadata.json")
    records: list[dict[str, Any]] = []
    for shard in range(SHARDS):
        eval_file = benchmark_shard(args, split, shard)
        expected = read_jsonl(eval_file)
        expected_indices = [int(row["eval_index"]) for row in expected]
        output = pass_output(args, target, split, repeat) / "shards" / f"shard_{shard:02d}"
        rows = read_jsonl(output / "records.jsonl")
        by_index = {int(row["eval_index"]): row for row in rows}
        if len(rows) != PROBLEMS_PER_SHARD or set(by_index) != set(expected_indices):
            raise RuntimeError(f"incomplete generation shard: {output}")
        policy = load_json(output / "generation_policy.json")
        expected_policy = generation_policy(
            target=target,
            dataset=split,
            repeat=repeat,
            checkpoint_identity=checkpoint,
            eval_file_sha256=sha256_file(eval_file),
            tokenizer_overlay_identity=str(overlay_metadata["overlay_identity"]),
        )
        if policy != expected_policy:
            raise RuntimeError(f"generation policy mismatch: {target}/{split}/{repeat}/{shard}")
        fingerprint = sha256_bytes(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode())
        for benchmark in expected:
            record = by_index[int(benchmark["eval_index"])]
            token_ids = record.get("generated_token_ids")
            generated_count = int(record.get("generated_token_count", -1))
            budget = int(record.get("effective_max_response_tokens", 0))
            finish_reason = str(record.get("finish_reason", ""))
            cap_hit = finish_reason == "length" or generated_count >= budget
            if (
                record.get("target") != target
                or record.get("dataset") != split
                or int(record.get("repeat", -1)) != repeat
                or record.get("checkpoint_manifest_sha256") != checkpoint
                or record.get("tokenizer_overlay_identity") != overlay_metadata["overlay_identity"]
                or record.get("policy_sha256") != fingerprint
                or budget < 1
                or int(record.get("rendered_prompt_tokens", 0)) < 1
                or int(record.get("rendered_prompt_tokens", 0)) + budget != 32768
                or int(record.get("total_context_limit", 0)) != 32768
                or int(record.get("seed", -1)) != sample_seed(repeat, int(benchmark["eval_index"]))
                or not isinstance(token_ids, list)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
                or generated_count != len(token_ids)
                or generated_count < 0
                or generated_count > budget
                or bool(record.get("cap_hit")) != cap_hit
                or sha256_bytes(str(record.get("response", "")).encode()) != record.get("response_sha256")
                or sha256_bytes(str(benchmark["prompt"]).encode()) != record.get("prompt_sha256")
                or record.get("problem_id") != benchmark["problem_id"]
                or int(record.get("answer", -1)) != benchmark["answer"]
            ):
                raise RuntimeError(
                    f"generation provenance mismatch: {target}/{split}/{repeat}/" f"{benchmark['eval_index']}"
                )
            records.append(record)
    if [int(row["eval_index"]) for row in records] != list(range(PROBLEMS_PER_SPLIT)):
        raise RuntimeError(f"merged generation order changed: {target}/{split}/{repeat}")
    return records


def percentile(values: list[int], percentile_value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def score_pass(args: argparse.Namespace, target: str, split: str, repeat: int) -> dict[str, Any]:
    benchmark = read_jsonl(benchmark_full(args, split))
    records = records_for_pass(args, target, split, repeat)
    scored = []
    for row, record in zip(benchmark, records, strict=True):
        trace = score_response(
            str(record["response"]),
            int(row["answer"]),
            finish_reason=record.get("finish_reason"),
            stop_token_id=(int(record["stop_reason"]) if isinstance(record.get("stop_reason"), int) else None),
            cap_hit=bool(record["cap_hit"]),
            problem_id=str(row["problem_id"]),
        )
        scored.append(
            {
                "generation": record,
                "gold_answer": int(row["answer"]),
                "problem_id": row["problem_id"],
                "scorer_trace": trace,
            }
        )
    correct = sum(bool(row["scorer_trace"]["is_correct"]) for row in scored)
    valid = sum(row["scorer_trace"]["official_extracted_source_span"] is not None for row in scored)
    caps = sum(bool(row["generation"]["cap_hit"]) for row in scored)
    cap_correct = sum(bool(row["scorer_trace"]["is_correct"]) for row in scored if row["generation"]["cap_hit"])
    parse_failures = sum(row["scorer_trace"]["official_decision_reason"] == "integer_parse_failure" for row in scored)
    lengths = [int(row["generation"]["generated_token_count"]) for row in scored]
    output = pass_output(args, target, split, repeat) / "merged"
    atomic_text(output / "scored.jsonl", jsonl_text(scored))
    summary = {
        "accuracy": correct / PROBLEMS_PER_SPLIT,
        "cap_hit_accuracy": cap_correct / caps if caps else 0.0,
        "cap_hit_correct": cap_correct,
        "cap_hit_rate": caps / PROBLEMS_PER_SPLIT,
        "cap_hits": caps,
        "correct": correct,
        "dataset": split,
        "finish_reasons": dict(sorted(Counter(str(row["generation"].get("finish_reason")) for row in scored).items())),
        "generated_length_mean": statistics.fmean(lengths),
        "generated_length_p50": percentile(lengths, 0.50),
        "generated_length_p95": percentile(lengths, 0.95),
        "parse_failure_count": parse_failures,
        "parse_failure_rate": parse_failures / PROBLEMS_PER_SPLIT,
        "repeat": repeat,
        "scorer_version": SCORER_VERSION,
        "target": target,
        "total": PROBLEMS_PER_SPLIT,
        "valid_box_count": valid,
        "valid_box_rate": valid / PROBLEMS_PER_SPLIT,
    }
    atomic_text(output / "summary.json", canonical_json(summary))
    atomic_text(output / "summary.sha256", sha256_file(output / "summary.json") + "\n")
    return summary


def mean_sd(values: list[float]) -> dict[str, Any]:
    if len(values) != REPEATS:
        raise ValueError(f"expected {REPEATS} values, found {len(values)}")
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def merge_target(args: argparse.Namespace) -> None:
    if len(args.scorer_commit) != 40 or any(character not in "0123456789abcdef" for character in args.scorer_commit):
        raise RuntimeError("scorer commit must be a full lowercase Git SHA-1")
    inventory = load_inventory(args)
    item = item_for(inventory, args.target)
    split_results: dict[str, Any] = {}
    aggregate_fields = (
        "accuracy",
        "valid_box_rate",
        "cap_hit_rate",
        "cap_hit_accuracy",
        "parse_failure_rate",
        "generated_length_mean",
        "generated_length_p50",
        "generated_length_p95",
    )
    for split in SPLITS:
        repeats = [score_pass(args, args.target, split, repeat) for repeat in range(REPEATS)]
        split_results[split] = {
            "aggregate": {field: mean_sd([float(row[field]) for row in repeats]) for field in aggregate_fields},
            "metric": "mean and sample standard deviation across three complete sampled pass@1 repeats",
            "repeats": repeats,
        }
    generation_contract = {
        "decoding": DECODING,
        "repeats": REPEATS,
        "response_budget": "32768 - rendered_prompt_tokens",
        "seed_formula": "1234 + 400 * repeat + eval_index",
        "stop_token_ids": sorted(STOP_TOKENS.values()),
        "target_context": TARGET_CONTEXT,
    }
    scorer_sha256 = sha256_file(EVAL_DIR / "aime_scorer.py")
    result = {
        "artifact_schema_version": 1,
        "benchmark": "sxiong/AIME-trajectory",
        "checkpoint_manifest_sha256": item["checkpoint_identity"],
        "dataset_hashes": {split: sha256_file(benchmark_full(args, split)) for split in SPLITS},
        "eval_config_sha256": sha256_file(EVAL_CONFIG),
        "generation_contract": generation_contract,
        "generation_contract_sha256": sha256_bytes(
            json.dumps(generation_contract, sort_keys=True, separators=(",", ":")).encode()
        ),
        "scorer_commit": args.scorer_commit,
        "scorer_sha256": scorer_sha256,
        "scorer_version": SCORER_VERSION,
        "source": "generated",
        "splits": split_results,
        "target": args.target,
        "training_condition": item.get("variant", args.target),
    }
    destination = result_root(args) / args.target
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "RESULTS.json": canonical_json(result),
        "RESULTS.md": (
            f"# {args.target} AIME generalization\n\n"
            f"- Held in: {split_results['held_in']['aggregate']['accuracy']['mean']:.2%} ± "
            f"{split_results['held_in']['aggregate']['accuracy']['sample_std']:.2%} SD\n"
            f"- Held out: {split_results['held_out_problem']['aggregate']['accuracy']['mean']:.2%} ± "
            f"{split_results['held_out_problem']['aggregate']['accuracy']['sample_std']:.2%} SD\n"
        ),
        "EVAL_POLICY.yaml": EVAL_CONFIG.read_text(encoding="utf-8"),
        "CHECKPOINT.json": canonical_json(load_json(manifest_path(args, item))),
        "PROVENANCE.json": canonical_json(
            {
                "artifact_schema_version": 1,
                "generated_results": str(eval_root(args) / args.target),
                "scorer_commit": args.scorer_commit,
                "scorer_sha256": scorer_sha256,
            }
        ),
    }
    for name, content in files.items():
        path = destination / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing incompatible compact result: {path}")
        atomic_text(path, content)
    print(f"AIME_TARGET_FINALIZED target={args.target}", flush=True)


def aggregate_results(
    args: argparse.Namespace,
    targets: tuple[str, ...],
    *,
    status: str,
    audit_name: str,
    result_name: str,
) -> None:
    if not set(REFERENCE_TARGETS).issubset(targets):
        raise RuntimeError("aggregate requires the base and 3K reference targets")
    if len(targets) != len(set(targets)) or not set(targets).issubset(TARGETS):
        raise RuntimeError("invalid aggregate target selection")
    results = {target: load_json(result_root(args) / target / "RESULTS.json") for target in targets}
    reference_result = results[BASE_TARGET]
    for target, result in results.items():
        if (
            result.get("target") != target
            or result.get("scorer_version") != SCORER_VERSION
            or result.get("scorer_commit") != reference_result.get("scorer_commit")
            or result.get("scorer_sha256") != reference_result.get("scorer_sha256")
            or result.get("eval_config_sha256") != reference_result.get("eval_config_sha256")
            or result.get("generation_contract_sha256") != reference_result.get("generation_contract_sha256")
            or result.get("dataset_hashes") != reference_result.get("dataset_hashes")
        ):
            raise RuntimeError(f"invalid aggregate result: {target}")

    rows = []
    for target in targets:
        result = results[target]
        split_values = {split: result["splits"][split]["aggregate"]["accuracy"] for split in SPLITS}
        gap = mean_sd(
            [
                float(out_value) - float(in_value)
                for in_value, out_value in zip(
                    split_values["held_in"]["values"],
                    split_values["held_out_problem"]["values"],
                    strict=True,
                )
            ]
        )
        deltas: dict[str, Any] = {}
        for reference_target in REFERENCE_TARGETS:
            for split in SPLITS:
                reference_values = results[reference_target]["splits"][split]["aggregate"]["accuracy"]["values"]
                deltas[f"{split}_minus_{reference_target}"] = mean_sd(
                    [
                        float(value) - float(reference_value)
                        for value, reference_value in zip(
                            split_values[split]["values"],
                            reference_values,
                            strict=True,
                        )
                    ]
                )
        diagnostics = {}
        for split in SPLITS:
            split_result = result["splits"][split]
            repeats = split_result["repeats"]
            diagnostics[split] = {
                "cap_hit_correct": sum(int(row["cap_hit_correct"]) for row in repeats),
                "cap_hits": sum(int(row["cap_hits"]) for row in repeats),
                "cap_hit_rate": split_result["aggregate"]["cap_hit_rate"],
                "generated_length_mean": split_result["aggregate"]["generated_length_mean"],
                "parse_failures": sum(int(row["parse_failure_count"]) for row in repeats),
                "valid_box_rate": split_result["aggregate"]["valid_box_rate"],
            }
        rows.append(
            {
                "deltas": deltas,
                "diagnostics": diagnostics,
                "generalization_gap": gap,
                "held_in_accuracy": split_values["held_in"],
                "held_out_accuracy": split_values["held_out_problem"],
                "target": target,
            }
        )

    value = {
        "artifact_schema_version": 1,
        "audited_at": now_iso(),
        "included_targets": list(targets),
        "pending_targets": [target for target in TARGETS if target not in targets],
        "results": rows,
        "scorer_commit": reference_result["scorer_commit"],
        "scorer_sha256": reference_result["scorer_sha256"],
        "scorer_version": SCORER_VERSION,
        "status": status,
    }
    atomic_text(control_root(args) / audit_name, canonical_json(value))
    atomic_text(result_root(args) / f"{result_name}.json", canonical_json(value))

    title_status = " — interim" if status == "interim" else ""
    lines = [
        f"# Qwen2.5-7B AIME distribution generalization{title_status} ({SCORER_VERSION})",
        "",
        "| Target | Held-in accuracy | Held-out accuracy | Held-out − held-in | Δ held-in vs base | Δ held-out vs base | Δ held-in vs 3K | Δ held-out vs 3K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:

        def fmt(metric: dict[str, Any], signed: bool = False) -> str:
            sign = "+" if signed else ""
            return f"{metric['mean']:{sign}.2%} ± {metric['sample_std']:.2%}"

        deltas = row["deltas"]
        lines.append(
            f"| {row['target']} | {fmt(row['held_in_accuracy'])} | "
            f"{fmt(row['held_out_accuracy'])} | {fmt(row['generalization_gap'], True)} | "
            f"{fmt(deltas['held_in_minus_base_qwen2_5_7b'], True)} | "
            f"{fmt(deltas['held_out_problem_minus_base_qwen2_5_7b'], True)} | "
            f"{fmt(deltas['held_in_minus_aime_correct_3000'], True)} | "
            f"{fmt(deltas['held_out_problem_minus_aime_correct_3000'], True)} |"
        )
    lines.extend(
        [
            "",
            "| Target | Split | Valid box | Cap hit | Correct cap hits | Parse failures | Mean generated tokens |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        for split in SPLITS:
            diagnostics = row["diagnostics"][split]
            lines.append(
                f"| {row['target']} | {split} | "
                f"{diagnostics['valid_box_rate']['mean']:.2%} ± {diagnostics['valid_box_rate']['sample_std']:.2%} | "
                f"{diagnostics['cap_hit_rate']['mean']:.2%} ± {diagnostics['cap_hit_rate']['sample_std']:.2%} | "
                f"{diagnostics['cap_hit_correct']}/{diagnostics['cap_hits']} | "
                f"{diagnostics['parse_failures']}/1200 | "
                f"{diagnostics['generated_length_mean']['mean']:.1f} |"
            )
    atomic_text(result_root(args) / f"{result_name}.md", "\n".join(lines) + "\n")
    print(
        f"AIME_RESULTS_AGGREGATED status={status} targets={len(targets)} "
        f"generations={len(targets) * len(SPLITS) * REPEATS * PROBLEMS_PER_SPLIT} "
        f"audit={control_root(args) / audit_name}",
        flush=True,
    )


def interim_audit(args: argparse.Namespace) -> None:
    aggregate_results(
        args,
        INTERIM_TARGETS,
        status="interim",
        audit_name="INTERIM_AUDIT.json",
        result_name="INTERIM_RESULTS",
    )


def audit(args: argparse.Namespace) -> None:
    aggregate_results(
        args,
        TARGETS,
        status="complete",
        audit_name="FINAL_AUDIT.json",
        result_name="RESULTS",
    )


def path_command(args: argparse.Namespace) -> None:
    inventory = load_inventory(args) if args.field in {"model", "tokenizer", "checkpoint-identity"} else None
    if args.field == "control":
        value: str | Path = control_root(args)
    elif args.field == "eval-root":
        value = eval_root(args)
    elif args.field == "result-root":
        value = result_root(args)
    elif args.field == "shard":
        value = benchmark_shard(args, args.dataset, args.shard)
    elif args.field == "smoke":
        value = benchmark_full(args, args.dataset).parent / "smoke.jsonl"
    elif args.field == "pass-output":
        value = pass_output(args, args.target, args.dataset, args.repeat)
    elif args.field == "model":
        assert inventory is not None
        value = model_path(args, item_for(inventory, args.target))
    elif args.field == "tokenizer":
        assert inventory is not None
        value = tokenizer_path(args, args.target, item_for(inventory, args.target))
    elif args.field == "checkpoint-identity":
        assert inventory is not None
        value = item_for(inventory, args.target)["checkpoint_identity"]
    else:
        raise AssertionError(args.field)
    print(value)


def validate_resubmission(args: argparse.Namespace) -> None:
    journal_path = Path(args.journal).resolve()
    failure_log = Path(args.failure_log).resolve()
    value = load_json(journal_path)
    if value.get("artifact_schema_version") not in {2, 3}:
        raise RuntimeError(f"unsupported prior submission schema: {journal_path}")
    arrays = value.get("array_jobs")
    finalizers = value.get("finalizer_jobs")
    if not isinstance(arrays, dict) or set(arrays) != set(TARGETS):
        raise RuntimeError("prior submission array inventory changed")
    if not isinstance(finalizers, dict) or set(finalizers) != set(TARGETS):
        raise RuntimeError("prior submission finalizer inventory changed")
    jobs = {
        str(value.get("preflight_job")),
        str(value.get("canary_job")),
        str(value.get("interim_job")),
        str(value.get("audit_job")),
        *(str(job) for job in arrays.values()),
        *(str(job) for job in finalizers.values()),
    }
    if len(jobs) != 30 or any(not job.isdigit() for job in jobs):
        raise RuntimeError(f"prior submission job inventory changed: {sorted(jobs)}")
    preflight = str(value["preflight_job"])
    if preflight not in failure_log.name:
        raise RuntimeError("failure log does not belong to the prior preflight")
    log_text = failure_log.read_text(encoding="utf-8", errors="replace")
    required = (
        "AttributeError: 'list' object has no attribute 'keys'",
        "_set_model_specific_special_tokens",
    )
    if not all(marker in log_text for marker in required):
        raise RuntimeError("prior preflight does not contain the approved tokenizer compatibility failure")
    generated = sorted(eval_root(args).glob("**/records.jsonl"))
    results = sorted(result_root(args).glob("**/*")) if result_root(args).is_dir() else []
    result_files = [path for path in results if path.is_file()]
    if generated or result_files:
        raise RuntimeError(
            "refusing to supersede an evaluation with generated/result artifacts: "
            f"records={generated} results={result_files}"
        )
    for job in sorted(jobs, key=int):
        print(job)


def record_submission(args: argparse.Namespace) -> None:
    arrays = dict(item.split("=", 1) for item in args.array_job)
    finalizers = dict(item.split("=", 1) for item in args.finalizer_job)
    if len(arrays) != len(TARGETS) or set(arrays) != set(TARGETS):
        raise RuntimeError("submission arrays do not cover every target exactly once")
    if len(finalizers) != len(TARGETS) or set(finalizers) != set(TARGETS):
        raise RuntimeError("submission finalizers do not cover every target exactly once")
    dependency_edges = [
        {"afterok": [args.preflight_job], "job": args.canary_job, "stage": "canary"},
        {"afterok": [args.canary_job], "job": arrays[BASE_TARGET], "stage": "base_generation"},
        {
            "afterok": [arrays[BASE_TARGET]],
            "job": finalizers[BASE_TARGET],
            "stage": "base_finalization",
        },
    ]
    predecessor = args.canary_job
    for target in PRIORITY_TARGETS:
        dependency_edges.extend(
            [
                {
                    "afterok": [predecessor],
                    "job": arrays[target],
                    "stage": "priority_generation",
                    "target": target,
                },
                {
                    "afterok": [arrays[target]],
                    "job": finalizers[target],
                    "stage": "priority_finalization",
                    "target": target,
                },
            ]
        )
        predecessor = finalizers[target]
    dependency_edges.append(
        {
            "afterok": [finalizers[BASE_TARGET], predecessor],
            "job": args.interim_job,
            "stage": "interim_aggregation",
        }
    )
    for target in DEFERRED_TARGETS:
        dependency_edges.extend(
            [
                {
                    "afterok": [args.interim_job],
                    "job": arrays[target],
                    "stage": "deferred_generation",
                    "target": target,
                },
                {
                    "afterok": [arrays[target]],
                    "job": finalizers[target],
                    "stage": "deferred_finalization",
                    "target": target,
                },
            ]
        )
    dependency_edges.append(
        {
            "afterok": [finalizers[target] for target in DEFERRED_TARGETS],
            "job": args.audit_job,
            "stage": "final_audit",
        }
    )
    if args.attempt < 1:
        raise RuntimeError("submission attempt must be positive")
    supersedes = None
    if args.attempt > 1:
        if not args.supersedes_journal or not args.failure_kind or not args.failure_log:
            raise RuntimeError("replacement submissions require complete supersession provenance")
        if not Path(args.supersedes_journal).is_file() or not Path(args.failure_log).is_file():
            raise RuntimeError("replacement submission provenance files are missing")
        supersedes = {
            "failure_kind": args.failure_kind,
            "failure_log": str(Path(args.failure_log).resolve()),
            "journal": str(Path(args.supersedes_journal).resolve()),
        }
    value = {
        "artifact_schema_version": 3,
        "array_jobs": arrays,
        "attempt": args.attempt,
        "audit_job": args.audit_job,
        "canary_job": args.canary_job,
        "dependency_edges": dependency_edges,
        "finalizer_jobs": finalizers,
        "git_commit": args.git_commit,
        "interim_job": args.interim_job,
        "preflight_job": args.preflight_job,
        "submitted_at": now_iso(),
        "supersedes": supersedes,
        "target_groups": {
            "base": [BASE_TARGET],
            "deferred": list(DEFERRED_TARGETS),
            "interim": list(INTERIM_TARGETS),
            "priority": list(PRIORITY_TARGETS),
        },
    }
    destination = control_root(args) / "submission.json"
    if destination.exists():
        raise FileExistsError(f"submission already recorded: {destination}")
    atomic_text(destination, canonical_json(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--held-in", required=True)
    prepare_parser.add_argument("--held-out", required=True)

    verify = subparsers.add_parser("verify-checkpoints")
    verify.add_argument("--full", action="store_true")
    verify.add_argument("--status")

    merge = subparsers.add_parser("merge-target")
    merge.add_argument("--target", required=True, choices=TARGETS)
    merge.add_argument("--scorer-commit", required=True)

    subparsers.add_parser("interim-audit")
    subparsers.add_parser("audit")

    path_parser = subparsers.add_parser("path")
    path_parser.add_argument(
        "--field",
        required=True,
        choices=[
            "control",
            "eval-root",
            "result-root",
            "shard",
            "smoke",
            "pass-output",
            "model",
            "tokenizer",
            "checkpoint-identity",
        ],
    )
    path_parser.add_argument("--target", choices=TARGETS)
    path_parser.add_argument("--dataset", choices=SPLITS)
    path_parser.add_argument("--repeat", type=int)
    path_parser.add_argument("--shard", type=int)

    record = subparsers.add_parser("record-submission")
    record.add_argument("--preflight-job", required=True)
    record.add_argument("--canary-job", required=True)
    record.add_argument("--array-job", action="append", default=[])
    record.add_argument("--finalizer-job", action="append", default=[])
    record.add_argument("--interim-job", required=True)
    record.add_argument("--audit-job", required=True)
    record.add_argument("--git-commit", required=True)
    record.add_argument("--attempt", required=True, type=int)
    record.add_argument("--supersedes-journal")
    record.add_argument("--failure-kind")
    record.add_argument("--failure-log")

    retry = subparsers.add_parser("validate-resubmission")
    retry.add_argument("--journal", required=True)
    retry.add_argument("--failure-log", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "verify-checkpoints":
        verify_checkpoints(args)
    elif args.command == "merge-target":
        merge_target(args)
    elif args.command == "interim-audit":
        interim_audit(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "path":
        path_command(args)
    elif args.command == "record-submission":
        record_submission(args)
    elif args.command == "validate-resubmission":
        validate_resubmission(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
