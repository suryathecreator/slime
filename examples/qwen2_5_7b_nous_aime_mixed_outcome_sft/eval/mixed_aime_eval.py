#!/usr/bin/env python3
"""Validate, score, aggregate, and record the mixed-outcome AIME evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.tokenizer_compat import prepare_overlay
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.hf_checkpoint_gate import validate_checkpoint

CONTRACT_ID = "2d556f01dbd853a1"
EVAL_NAME = "aime_mixed_outcome_sampled_v1"
TARGETS = (
    "base_qwen2_5_7b",
    "aime24_correct_3000",
    "aime24_incorrect_3000",
    "aime25_correct_3000",
    "aime25_incorrect_3000",
)
DRAWS = 16
PROMPTS = 60
EVAL_CONTRACT_SHA256 = "f49e6b6337ed2b2186b0412cd874583e79d96d89c8b80a91b13d37f0a6ac8b92"
SCORER_SHA256 = "c6b10ee5a5667a21df095f9ef87a04c2499cf8dbe56cff11f3bf9836d8262256"
DATASET_SHA256 = {
    "aime24": "0c838ee6351d9636a7fdb62dbf9a1123c3fbf1cd13e61325aa892ac4343f6538",
    "aime25": "195dd659d4c20a3900875cfe7e83acc9a69c16a7c73ad6d6eb5f61c9fd30f70e",
}
STOP_TOKENS = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
REQUESTED_ARTIFACTS = (
    "handoff/EVAL_HANDOFF.md",
    "handoff/eval_contract.json",
    "handoff/comparison_sources.json",
    "data/eval/aime24_30.jsonl",
    "data/eval/aime25_30.jsonl",
    "TRAINING_STATUS.json",
    "config/experiment_contract.json",
    "handoff/checkpoint_sources.json",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def experiment_root(args: argparse.Namespace) -> Path:
    return args.experiment_root.resolve()


def contract_id(args: argparse.Namespace) -> str:
    return sha256_file(experiment_root(args) / "config" / "experiment_contract.json")[:16]


def final_iteration(args: argparse.Namespace) -> int:
    contract = load_json(experiment_root(args) / "config" / "experiment_contract.json")
    value = int(contract["training"]["final_iteration"])
    if value not in {46, 187}:
        raise RuntimeError(f"unsupported training final iteration: {value}")
    return value


def eval_root(args: argparse.Namespace) -> Path:
    return experiment_root(args) / "outputs" / "eval" / EVAL_NAME


def result_root(args: argparse.Namespace) -> Path:
    return experiment_root(args) / "results" / EVAL_NAME


def control_root(args: argparse.Namespace) -> Path:
    return eval_root(args) / "_control"


def comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    inventory = load_json(experiment_root(args) / "handoff" / "comparison_sources.json")
    targets = inventory.get("targets")
    if (
        inventory.get("contract_hash") != contract_id(args)
        or inventory.get("comparison_checkpoint_count") != len(TARGETS)
        or not isinstance(targets, list)
        or tuple(row.get("id") for row in targets) != TARGETS
    ):
        raise RuntimeError("comparison inventory does not match the five-target contract")
    return targets


def target_item(args: argparse.Namespace, target: str) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    return next(row for row in comparison(args) if row["id"] == target)


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe relative path: {value}")
    return path


def target_paths(args: argparse.Namespace, target: str) -> tuple[Path, Path, str]:
    item = target_item(args, target)
    relative_root = safe_relative(str(item["remote_experiment_relative"]))
    declared_root = args.repo_root.resolve() / relative_root
    if target != "base_qwen2_5_7b" and declared_root.resolve() != experiment_root(args):
        raise RuntimeError(f"trained checkpoint root mismatch: {target}")
    model = declared_root / safe_relative(str(item["remote_checkpoint_relative"]))
    manifest = declared_root / safe_relative(str(item["remote_manifest_relative"]))
    identity = str(item["checkpoint_identity"])
    if len(identity) != 64:
        raise RuntimeError(f"invalid checkpoint identity: {target}")
    return model, manifest, identity


def eval_files(args: argparse.Namespace) -> tuple[Path, Path]:
    root = experiment_root(args) / "data" / "eval"
    return root / "aime24_30.jsonl", root / "aime25_30.jsonl"


def normalized_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    files = eval_files(args)
    for year_name, year, path, start in zip(("aime24", "aime25"), (2024, 2025), files, (0, 30), strict=True):
        if sha256_file(path) != DATASET_SHA256[year_name]:
            raise RuntimeError(f"dataset hash mismatch: {path}")
        rows = read_jsonl(path)
        if len(rows) != 30:
            raise RuntimeError(f"{year_name} must have 30 rows")
        for local_index, row in enumerate(rows):
            index = start + local_index
            query = row.get("query")
            answer = row.get("answer")
            expected_problem = f"{year_name}:{local_index}"
            if row.get("global_eval_index") != index or row.get("problem_id") != expected_problem:
                raise RuntimeError(f"noncanonical row identity: {year_name}/{local_index}")
            if row.get("doc_id") != str(local_index) or row.get("year") != year:
                raise RuntimeError(f"noncanonical source identity: {expected_problem}")
            if not isinstance(query, str) or sha256_text(query) != row.get("query_sha256"):
                raise RuntimeError(f"query hash mismatch: {expected_problem}")
            if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer <= 999:
                raise RuntimeError(f"invalid AIME answer: {expected_problem}")
            if row.get("split") not in {"held_in", "held_out"}:
                raise RuntimeError(f"invalid split: {expected_problem}")
            result.append(dict(row))
    if [row["global_eval_index"] for row in result] != list(range(PROMPTS)):
        raise RuntimeError("global evaluation indices are not exactly 0..59")
    for year in (2024, 2025):
        counts = {
            split: sum(row["year"] == year and row["split"] == split for row in result)
            for split in ("held_in", "held_out")
        }
        if counts != {"held_in": 10, "held_out": 20}:
            raise RuntimeError(f"split counts changed for {year}: {counts}")
    if len({row["problem_id"] for row in result}) != PROMPTS:
        raise RuntimeError("duplicate problem IDs")
    return result


def validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    root = experiment_root(args)
    for relative in REQUESTED_ARTIFACTS:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"missing requested artifact: {relative}")
    experiment_contract = root / "config" / "experiment_contract.json"
    if sha256_file(experiment_contract)[:16] != contract_id(args):
        raise RuntimeError("experiment contract hash/path mismatch")
    contract = load_json(root / "handoff" / "eval_contract.json")
    if sha256_file(root / "handoff" / "eval_contract.json") != EVAL_CONTRACT_SHA256:
        raise RuntimeError("evaluation contract file hash changed")
    expected = {
        "draws_per_prompt": DRAWS,
        "generation_count": len(TARGETS) * PROMPTS * DRAWS,
        "max_total_context_tokens": 32768,
        "cap_hits_scored": True,
        "checkpoints": list(TARGETS),
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(f"eval contract changed at {key}")
    if contract.get("sampling") != {
        "n": 1,
        "seed": "1234 + 60 * draw_index + global_eval_index",
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.8,
    }:
        raise RuntimeError("sampling contract changed")
    scorer = contract.get("scorer", {})
    scorer_path = args.repo_root.resolve() / str(scorer.get("path_in_repo", ""))
    if scorer.get("sha256") != SCORER_SHA256 or sha256_file(scorer_path) != SCORER_SHA256:
        raise RuntimeError("contracted scorer hash mismatch")
    sources = contract.get("evaluation_sources")
    if not isinstance(sources, list) or [(row.get("year"), row.get("rows"), row.get("sha256")) for row in sources] != [
        ("aime24", 30, DATASET_SHA256["aime24"]),
        ("aime25", 30, DATASET_SHA256["aime25"]),
    ]:
        raise RuntimeError("evaluation source contract changed")
    return contract


def validate_checkpoint_files(model: Path, manifest_path: Path, identity: str, full: bool) -> dict[str, Any]:
    if not model.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing checkpoint or manifest: {model}")
    validate_checkpoint(model)
    manifest = load_json(manifest_path)
    if manifest.get("checkpoint_manifest_sha256") != identity:
        raise RuntimeError(f"checkpoint identity mismatch: {model}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"empty checkpoint manifest: {manifest_path}")
    expected = {str(item["name"]): (int(item["bytes"]), str(item["sha256"])) for item in files}
    actual = {path.relative_to(model).as_posix(): path.stat().st_size for path in model.rglob("*") if path.is_file()}
    if actual != {name: size_hash[0] for name, size_hash in expected.items()}:
        raise RuntimeError(f"checkpoint size inventory mismatch: {model}")
    if full:
        for name, (_, expected_hash) in expected.items():
            if sha256_file(model / name) != expected_hash:
                raise RuntimeError(f"checkpoint file hash mismatch: {model / name}")
    return {"files": len(expected), "identity": identity, "model": str(model), "full_hashes": full}


def scorer_function(args: argparse.Namespace) -> Callable[..., dict[str, Any]]:
    path = args.repo_root.resolve() / "examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval/aime_scorer.py"
    if sha256_file(path) != SCORER_SHA256:
        raise RuntimeError("refusing scorer with unexpected hash")
    spec = importlib.util.spec_from_file_location("contracted_aime_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.SCORER_VERSION != "aime_last_boxed_integer_scorer_v1":
        raise RuntimeError("scorer version changed")
    return module.score_response


def preflight(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    contract = validate_contract(args)
    rows = normalized_rows(args)
    sources = load_json(experiment_root(args) / "handoff" / "checkpoint_sources.json")
    if (
        sources.get("contract_hash") != contract_id(args)
        or sources.get("checkpoint_count") != 4
        or sources.get("training_datasets_transferred") is not False
        or tuple(row.get("id") for row in sources.get("checkpoints", [])) != TARGETS[1:]
    ):
        raise RuntimeError("checkpoint transfer inventory changed")
    transferred = {row["id"]: row for row in sources["checkpoints"]}
    for row in comparison(args)[1:]:
        source = transferred[row["id"]]
        for field in (
            "checkpoint_identity",
            "remote_checkpoint_relative",
            "remote_manifest_relative",
            "source_checkpoint",
            "source_manifest",
        ):
            if source.get(field) != row.get(field):
                raise RuntimeError(f"checkpoint/comparison inventory mismatch: {row['id']}/{field}")
    status = load_json(experiment_root(args) / "TRAINING_STATUS.json")
    if (
        status.get("contract_hash") != contract_id(args)
        or status.get("trained_checkpoints") != 4
        or status.get("final_iterations") != {target: final_iteration(args) for target in TARGETS[1:]}
    ):
        raise RuntimeError("training status is incomplete")

    checkpoint_status: dict[str, Any] = {}
    prompt_id_reference: dict[int, list[int]] = {}
    overlays = control_root(args) / "tokenizer_overlays"
    for target in TARGETS:
        print(f"MIXED_AIME_PREFLIGHT_TARGET_START target={target}", flush=True)
        model, manifest, identity = target_paths(args, target)
        if status.get("checkpoint_identities", {}).get(target) != identity:
            raise RuntimeError(f"training/comparison identity mismatch: {target}")
        checkpoint_status[target] = validate_checkpoint_files(model, manifest, identity, args.full)
        overlay_path, overlay_metadata = prepare_overlay(overlays, target, model, identity)
        tokenizer = AutoTokenizer.from_pretrained(overlay_path, trust_remote_code=True, use_fast=True)
        actual_stops = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in STOP_TOKENS}
        if actual_stops != STOP_TOKENS or int(tokenizer.eos_token_id) != 151643:
            raise RuntimeError(f"stop-token mismatch: {target}")
        lengths: list[int] = []
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["query"]}], tokenize=False, add_generation_prompt=True
            )
            expected_prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            if not rendered.startswith(expected_prefix) or not rendered.endswith(
                "<|im_end|>\n<|im_start|>assistant\n"
            ):
                raise RuntimeError(f"chat template rendering changed: {target}/{row['problem_id']}")
            ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
            if not ids or len(ids) >= 32768:
                raise RuntimeError(f"invalid rendered prompt length: {target}/{row['problem_id']}")
            index = int(row["global_eval_index"])
            if target == TARGETS[0]:
                prompt_id_reference[index] = ids
            elif ids != prompt_id_reference[index]:
                raise RuntimeError(f"tokenizer overlay changed prompt IDs: {target}/{row['problem_id']}")
            lengths.append(len(ids))
        checkpoint_status[target]["overlay_identity"] = overlay_metadata["overlay_identity"]
        checkpoint_status[target]["prompt_tokens"] = {"min": min(lengths), "max": max(lengths)}
        print(
            f"MIXED_AIME_PREFLIGHT_TARGET_OK target={target} " f"prompt_tokens={min(lengths)}-{max(lengths)}",
            flush=True,
        )

    artifact = {
        "artifact_schema_version": 1,
        "checkpoints": checkpoint_status,
        "contract_id": contract_id(args),
        "contract_sha256": sha256_file(experiment_root(args) / "handoff" / "eval_contract.json"),
        "datasets_sha256": DATASET_SHA256,
        "generation_count": contract["generation_count"],
        "prompt_count": len(rows),
        "scorer_sha256": SCORER_SHA256,
        "status": "valid",
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_text(control_root(args) / "preflight.json", canonical_json(artifact))
    print(f"MIXED_AIME_PREFLIGHT_OK targets={len(TARGETS)} prompts={len(rows)} full={args.full}")


def draw_output(args: argparse.Namespace, target: str, draw: int) -> Path:
    if target not in TARGETS or draw not in range(DRAWS):
        raise ValueError(f"invalid draw coordinate: {target}/{draw}")
    return eval_root(args) / target / f"draw_{draw:02d}"


def validate_generation_policy(
    args: argparse.Namespace,
    target: str,
    draw: int,
    checkpoint_identity: str,
    tokenizer_overlay_identity: str,
) -> str:
    path = draw_output(args, target, draw) / "generation_policy.json"
    policy = load_json(path)
    expected = {
        "artifact_schema_version": 1,
        "checkpoint_identity": checkpoint_identity,
        "contract_sha256": EVAL_CONTRACT_SHA256,
        "datasets_sha256": [DATASET_SHA256["aime24"], DATASET_SHA256["aime25"]],
        "decoding": {"temperature": 0.7, "top_p": 0.8, "top_k": -1, "n": 1},
        "draw_index": draw,
        "response_budget": "32768 - rendered_prompt_tokens",
        "seed_formula": "1234 + 60 * draw_index + global_eval_index",
        "stop_token_ids": [151643, 151645],
        "target": target,
        "target_context": 32768,
        "tokenizer_overlay_identity": tokenizer_overlay_identity,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise RuntimeError(f"generation policy mismatch: {target}/{draw}/{key}")
    prompt = policy.get("prompt")
    if prompt != {
        "add_generation_prompt": True,
        "input_key": "query",
        "messages": "one user message containing query verbatim",
        "system_inserted_by_template": "You are a helpful assistant.",
    }:
        raise RuntimeError(f"generation prompt policy mismatch: {target}/{draw}")
    return sha256_text(json.dumps(policy, sort_keys=True, separators=(",", ":")))


def score_target(args: argparse.Namespace) -> None:
    score_response = scorer_function(args)
    target = args.target
    model, _, expected_identity = target_paths(args, target)
    _, overlay_metadata = prepare_overlay(control_root(args) / "tokenizer_overlays", target, model, expected_identity)
    expected_overlay_identity = str(overlay_metadata["overlay_identity"])
    canonical = {int(row["global_eval_index"]): row for row in normalized_rows(args)}
    scored: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for draw in range(DRAWS):
        policy_sha256 = validate_generation_policy(args, target, draw, expected_identity, expected_overlay_identity)
        path = draw_output(args, target, draw) / "records.jsonl"
        rows = read_jsonl(path)
        if len(rows) != PROMPTS:
            raise RuntimeError(f"draw is incomplete: {target}/{draw} ({len(rows)})")
        for record in rows:
            index = int(record.get("global_eval_index", -1))
            coordinate = (draw, index)
            source = canonical.get(index, {})
            if (
                coordinate in seen
                or record.get("target") != target
                or record.get("draw_index") != draw
                or record.get("checkpoint_identity") != expected_identity
                or record.get("seed") != 1234 + PROMPTS * draw + index
                or record.get("policy_sha256") != policy_sha256
                or record.get("tokenizer_overlay_identity") != expected_overlay_identity
                or record.get("answer") != source.get("answer")
                or record.get("problem_id") != source.get("problem_id")
                or record.get("query_sha256") != source.get("query_sha256")
                or record.get("split") != source.get("split")
                or record.get("year") != source.get("year")
                or sha256_text(str(record.get("response", ""))) != record.get("response_sha256")
            ):
                raise RuntimeError(f"invalid generation coordinate: {target}/{coordinate}")
            seen.add(coordinate)
            trace = score_response(
                str(record["response"]),
                int(record["answer"]),
                finish_reason=str(record.get("finish_reason") or ""),
                stop_token_id=record.get("stop_reason") if isinstance(record.get("stop_reason"), int) else None,
                cap_hit=bool(record["cap_hit"]),
                problem_id=str(record["problem_id"]),
            )
            scored.append(
                {
                    "answer": int(record["answer"]),
                    "cap_hit": bool(record["cap_hit"]),
                    "draw_index": draw,
                    "finish_reason": record.get("finish_reason"),
                    "generated_token_count": int(record["generated_token_count"]),
                    "global_eval_index": index,
                    "is_correct": bool(trace["is_correct"]),
                    "is_scorable": bool(trace["is_scorable"]),
                    "official_candidate_raw": trace["official_candidate_raw"],
                    "official_decision_reason": trace["official_decision_reason"],
                    "official_extracted_source_span": trace["official_extracted_source_span"],
                    "parsed_integer": trace["parsed_integer"],
                    "problem_id": str(record["problem_id"]),
                    "rendered_prompt_tokens": int(record["rendered_prompt_tokens"]),
                    "response_sha256": str(record["response_sha256"]),
                    "scorer_version": trace["scorer_version"],
                    "seed": int(record["seed"]),
                    "split": str(record["split"]),
                    "stop_reason": record.get("stop_reason"),
                    "target": target,
                    "trace_schema_version": trace["trace_schema_version"],
                    "year": int(record["year"]),
                }
            )
    if seen != {(draw, index) for draw in range(DRAWS) for index in range(PROMPTS)}:
        raise RuntimeError(f"target has missing generation coordinates: {target}")
    scored.sort(key=lambda row: (row["draw_index"], row["global_eval_index"]))
    output = result_root(args) / "targets" / target / "scores.jsonl"
    atomic_text(output, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored))
    summary = {name: metrics([row for row in scored if selector(row)]) for name, selector in slice_selectors().items()}
    atomic_text(output.parent / "summary.json", canonical_json({"target": target, "slices": summary}))
    print(f"MIXED_AIME_TARGET_FINALIZED target={target} scores={len(scored)}")


def slice_selectors() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "aime24_held_in": lambda row: row["year"] == 2024 and row["split"] == "held_in",
        "aime24_held_out": lambda row: row["year"] == 2024 and row["split"] == "held_out",
        "aime24_all": lambda row: row["year"] == 2024,
        "aime25_held_in": lambda row: row["year"] == 2025 and row["split"] == "held_in",
        "aime25_held_out": lambda row: row["year"] == 2025 and row["split"] == "held_out",
        "aime25_all": lambda row: row["year"] == 2025,
        "all_60": lambda row: True,
    }


def sample_se(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot aggregate an empty slice")
    correct = [float(bool(row["is_correct"])) for row in rows]
    lengths = [int(row["generated_token_count"]) for row in rows]
    return {
        "accuracy": statistics.fmean(correct),
        "accuracy_mc_se": sample_se(correct),
        "cap_hit_rate": statistics.fmean(float(bool(row["cap_hit"])) for row in rows),
        "completion_tokens": {
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        },
        "correct": sum(bool(row["is_correct"]) for row in rows),
        "n": len(rows),
        "valid_box_rate": statistics.fmean(float(bool(row["is_scorable"])) for row in rows),
    }


def paired_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_map = {(row["draw_index"], row["global_eval_index"]): row for row in left}
    right_map = {(row["draw_index"], row["global_eval_index"]): row for row in right}
    if left_map.keys() != right_map.keys():
        raise RuntimeError("paired delta coordinates differ")
    values = [
        float(bool(right_map[key]["is_correct"])) - float(bool(left_map[key]["is_correct"]))
        for key in sorted(left_map)
    ]
    return {"delta": statistics.fmean(values), "delta_mc_se": sample_se(values), "n": len(values)}


def primary_table(records: dict[str, list[dict[str, Any]]], year: int) -> list[dict[str, Any]]:
    other = 2025 if year == 2024 else 2024
    prefix = f"aime{str(year)[-2:]}"
    correct_target = f"{prefix}_correct_3000"
    incorrect_target = f"{prefix}_incorrect_3000"
    rows = [
        (f"AIME {year} held-in", f"{prefix}_held_in"),
        (f"AIME {year} held-out", f"{prefix}_held_out"),
        (f"AIME {year} all-30", f"{prefix}_all"),
        (f"AIME {other} cross-year", f"aime{str(other)[-2:]}_all"),
    ]
    selectors = slice_selectors()
    result: list[dict[str, Any]] = []
    for label, slice_name in rows:
        selected = {
            target: [row for row in values if selectors[slice_name](row)] for target, values in records.items()
        }
        result.append(
            {
                "base": metrics(selected["base_qwen2_5_7b"]),
                "correct": metrics(selected[correct_target]),
                "incorrect": metrics(selected[incorrect_target]),
                "incorrect_minus_correct": paired_delta(selected[correct_target], selected[incorrect_target]),
                "label": label,
                "slice": slice_name,
            }
        )
    return result


def fmt_rate(value: float, se: float | None = None) -> str:
    base = f"{100 * value:.2f}%"
    return base if se is None else f"{base} ± {100 * se:.2f}%"


def markdown_table(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"## {title}",
        "",
        "| Slice | Base | Correct-trained | Incorrect-trained | Incorrect − correct |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["incorrect_minus_correct"]
        lines.append(
            "| {label} | {base} | {correct} | {incorrect} | {delta} |".format(
                label=row["label"],
                base=fmt_rate(row["base"]["accuracy"], row["base"]["accuracy_mc_se"]),
                correct=fmt_rate(row["correct"]["accuracy"], row["correct"]["accuracy_mc_se"]),
                incorrect=fmt_rate(row["incorrect"]["accuracy"], row["incorrect"]["accuracy_mc_se"]),
                delta=fmt_rate(delta["delta"], delta["delta_mc_se"]),
            )
        )
    return "\n".join(lines)


def markdown_diagnostics(diagnostics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "## All-60 diagnostics",
        "",
        "Valid box means the contracted scorer parsed a complete boxed AIME integer.",
        "",
        "| Target | Valid box | Cap hit | Mean tokens | Std | P50 | P90 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        summary = diagnostics[target]["all_60"]
        lengths = summary["completion_tokens"]
        lines.append(
            "| {target} | {valid} | {cap} | {mean:.1f} | {std:.1f} | {p50:.1f} | "
            "{p90:.1f} | {p95:.1f} | {p99:.1f} | {maximum} |".format(
                target=target,
                valid=fmt_rate(summary["valid_box_rate"]),
                cap=fmt_rate(summary["cap_hit_rate"]),
                mean=lengths["mean"],
                std=lengths["std"],
                p50=lengths["p50"],
                p90=lengths["p90"],
                p95=lengths["p95"],
                p99=lengths["p99"],
                maximum=lengths["max"],
            )
        )
    return "\n".join(lines)


def audit(args: argparse.Namespace) -> None:
    records: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        path = result_root(args) / "targets" / target / "scores.jsonl"
        rows = read_jsonl(path)
        if len(rows) != PROMPTS * DRAWS:
            raise RuntimeError(f"target score count changed: {target}/{len(rows)}")
        records[target] = rows
    tables = {"aime24_trained": primary_table(records, 2024), "aime25_trained": primary_table(records, 2025)}
    diagnostics = {
        target: {name: metrics([row for row in rows if selector(row)]) for name, selector in slice_selectors().items()}
        for target, rows in records.items()
    }
    report = {
        "aggregation": "mean of all independent correctness indicators; expected single-sample accuracy",
        "artifact_schema_version": 1,
        "contract_id": contract_id(args),
        "diagnostics": diagnostics,
        "draws_per_prompt": DRAWS,
        "generation_count": sum(len(rows) for rows in records.values()),
        "monte_carlo_se": "sample standard deviation divided by sqrt(n); paired differences for deltas",
        "scorer_sha256": SCORER_SHA256,
        "tables": tables,
    }
    root = result_root(args)
    atomic_text(root / "FINAL_AUDIT.json", canonical_json(report))
    markdown = (
        "\n\n".join(
            (
                "# Nous AIME mixed-outcome evaluation\n\nEvery completion, including cap hits, is scored independently.",
                markdown_table("AIME 2024-trained pair", tables["aime24_trained"]),
                markdown_table("AIME 2025-trained pair", tables["aime25_trained"]),
                markdown_diagnostics(diagnostics),
            )
        )
        + "\n"
    )
    atomic_text(root / "RESULTS.md", markdown)
    print(f"MIXED_AIME_AUDIT_COMPLETE generations={report['generation_count']}")


def path_command(args: argparse.Namespace) -> None:
    if args.field == "control":
        value: str | Path = control_root(args)
    elif args.field == "eval-root":
        value = eval_root(args)
    elif args.field == "result-root":
        value = result_root(args)
    elif args.field in {"model", "manifest", "identity", "tokenizer"}:
        model, manifest, identity = target_paths(args, args.target)
        if args.field == "model":
            value = model
        elif args.field == "manifest":
            value = manifest
        elif args.field == "identity":
            value = identity
        else:
            value, _ = prepare_overlay(control_root(args) / "tokenizer_overlays", args.target, model, identity)
    elif args.field == "draw-output":
        value = draw_output(args, args.target, args.draw)
    else:
        raise AssertionError(args.field)
    print(value)


def record_submission(args: argparse.Namespace) -> None:
    arrays = dict(value.split("=", 1) for value in args.array_job)
    finalizers = dict(value.split("=", 1) for value in args.finalizer_job)
    if tuple(arrays) != TARGETS or tuple(finalizers) != TARGETS:
        raise RuntimeError("submission job assignments are not in canonical target order")
    edges: list[dict[str, Any]] = [{"afterok": [args.preflight_job], "job": args.canary_job, "stage": "canary"}]
    for target in TARGETS:
        edges.extend(
            (
                {"afterok": [args.canary_job], "job": arrays[target], "stage": "generation", "target": target},
                {"afterok": [arrays[target]], "job": finalizers[target], "stage": "finalization", "target": target},
            )
        )
    edges.append({"afterok": list(finalizers.values()), "job": args.audit_job, "stage": "final_audit"})
    metadata = {
        "array_jobs": arrays,
        "artifact_schema_version": 1,
        "audit_job": args.audit_job,
        "canary_job": args.canary_job,
        "contract_id": contract_id(args),
        "contract_sha256": sha256_file(experiment_root(args) / "handoff" / "eval_contract.json"),
        "dependency_edges": edges,
        "draws_per_prompt": DRAWS,
        "evaluator_git_commit": args.git_commit,
        "finalizer_jobs": finalizers,
        "generation_count": len(TARGETS) * PROMPTS * DRAWS,
        "log_root": str(control_root(args) / "slurm_logs"),
        "output_root": str(eval_root(args)),
        "preflight_job": args.preflight_job,
        "resources": {
            "generation": {"array": "0-15", "cpus": 8, "gpus": "h200:1", "memory": "96G", "time": "02:00:00"},
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": -1, "n": 1},
        },
        "scorer_sha256": SCORER_SHA256,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "targets": list(TARGETS),
    }
    journal = control_root(args) / "submission.json"
    if journal.exists():
        raise RuntimeError(f"submission journal already exists: {journal}")
    atomic_text(journal, canonical_json(metadata))
    print(journal)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--experiment-root", type=Path, required=True)
    commands = result.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--full", action="store_true")
    preflight_parser.set_defaults(func=preflight)
    path_parser = commands.add_parser("path")
    path_parser.add_argument(
        "--field",
        required=True,
        choices=("control", "eval-root", "result-root", "model", "manifest", "identity", "tokenizer", "draw-output"),
    )
    path_parser.add_argument("--target", choices=TARGETS)
    path_parser.add_argument("--draw", type=int)
    path_parser.set_defaults(func=path_command)
    finalize_parser = commands.add_parser("finalize-target")
    finalize_parser.add_argument("--target", required=True, choices=TARGETS)
    finalize_parser.set_defaults(func=score_target)
    audit_parser = commands.add_parser("audit")
    audit_parser.set_defaults(func=audit)
    record = commands.add_parser("record-submission")
    record.add_argument("--preflight-job", required=True)
    record.add_argument("--canary-job", required=True)
    record.add_argument("--array-job", action="append", required=True)
    record.add_argument("--finalizer-job", action="append", required=True)
    record.add_argument("--audit-job", required=True)
    record.add_argument("--git-commit", required=True)
    record.set_defaults(func=record_submission)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.repo_root.resolve() not in experiment_root(args).parents:
        raise RuntimeError("experiment root must be inside the repository")
    args.func(args)


if __name__ == "__main__":
    main()
