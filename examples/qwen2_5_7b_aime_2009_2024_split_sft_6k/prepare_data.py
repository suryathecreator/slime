#!/usr/bin/env python3
"""Build four matched 6K Qwen2.5 SFT variants from the pinned Qwen3 AIME corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.data_utils import (
    PROMPT_TEMPLATE,
    SPLITS,
    VARIANTS,
    atomic_json,
    atomic_jsonl,
    build_record,
    expand_balanced,
    length_summary,
    ordered_values_sha256,
    partition_problem_ids,
    select_correct_control,
    sha256_file,
    sha256_text,
    tokenizer_control_inventory,
)

REQUIRED_FIELDS = {
    "accepted_answer_integers",
    "cap_hit",
    "complete_box_found",
    "contest",
    "contract_sha256",
    "decision_reason",
    "draw_index",
    "effective_max_new_tokens",
    "eval_index",
    "extracted_answer_integer",
    "extracted_answer_raw",
    "finish_reason",
    "generation_parameters_json",
    "gold_answer_canonical",
    "gold_answer_raw",
    "has_asy",
    "input_tokens",
    "is_correct",
    "is_scorable",
    "matched_gold_integer",
    "messages_json",
    "model_id",
    "model_revision",
    "output_tokens",
    "part",
    "problem",
    "problem_id",
    "problem_number",
    "prompt",
    "rendered_prompt",
    "rendered_prompt_sha256",
    "response",
    "response_sha256",
    "sample_id",
    "scorer_trace_json",
    "scorer_version",
    "seed",
    "source_dataset",
    "source_kind",
    "source_record_sha256",
    "source_revision",
    "source_url",
    "stop_condition",
    "stop_token_id",
    "year",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_tokenizer(model: Path, contract: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    for filename, expected in contract["base_model"]["tokenizer_files"].items():
        if sha256_file(model / filename) != expected:
            raise ValueError(f"base tokenizer hash drift: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    expected_controls = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}
    inventory = tokenizer_control_inventory(tokenizer)
    for token, token_id in expected_controls.items():
        if inventory["added_vocabulary"].get(token) != token_id:
            raise ValueError(f"tokenizer control drift for {token}")
    return tokenizer


def validate_source_row(row: dict[str, Any], draw_index: int) -> None:
    if set(row) != REQUIRED_FIELDS:
        raise ValueError(
            f"source schema drift: missing={sorted(REQUIRED_FIELDS - set(row))} "
            f"extra={sorted(set(row) - REQUIRED_FIELDS)}"
        )
    if int(row["draw_index"]) != draw_index:
        raise ValueError("draw-index/file mismatch")
    if row["sample_id"] != f"{row['problem_id']}:draw-{draw_index:02d}":
        raise ValueError("sample_id drift")
    if row["model_id"] != "Qwen/Qwen3-8B" or row["scorer_version"] != "aime_last_boxed_integer_scorer_v1":
        raise ValueError("source model/scorer identity drift")
    if sha256_text(row["response"]) != row["response_sha256"]:
        raise ValueError("source response hash drift")
    if sha256_text(row["rendered_prompt"]) != row["rendered_prompt_sha256"]:
        raise ValueError("source rendered-prompt hash drift")
    expected_prompt = PROMPT_TEMPLATE.replace("{problem}", row["problem"])
    if row["prompt"] != expected_prompt:
        raise ValueError("source prompt differs from pinned generation query")
    if json.loads(row["messages_json"]) != [{"role": "user", "content": row["prompt"]}]:
        raise ValueError("source messages are not exactly one user message with no system")
    scorer_trace = json.loads(row["scorer_trace_json"])
    if bool(scorer_trace["is_correct"]) != bool(row["is_correct"]):
        raise ValueError("stored scorer trace/correctness drift")


def load_source(
    source_dir: Path,
    contract: dict[str, Any],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    problem_invariants: dict[str, dict[str, Any]] = {}
    seen_samples: set[str] = set()
    expected_files = contract["source_dataset"]["files"]
    for draw_index in range(16):
        filename = f"draw-{draw_index:02d}.parquet"
        path = source_dir / filename
        expected = expected_files[filename]
        if sha256_file(path) != expected:
            raise ValueError(f"pinned source hash drift: {filename}")
        shard = pq.read_table(path).to_pylist()
        if len(shard) != 480:
            raise ValueError(f"{filename} must contain exactly 480 rows")
        for row in shard:
            validate_source_row(row, draw_index)
            sample_id = str(row["sample_id"])
            if sample_id in seen_samples:
                raise ValueError(f"duplicate source sample: {sample_id}")
            seen_samples.add(sample_id)
            problem_id = str(row["problem_id"])
            invariant = {
                "accepted_answer_integers": [int(value) for value in row["accepted_answer_integers"]],
                "contest": str(row["contest"]),
                "eval_index": int(row["eval_index"]),
                "gold_answer_canonical": str(row["gold_answer_canonical"]),
                "part": str(row["part"]),
                "problem": str(row["problem"]),
                "problem_id": problem_id,
                "problem_number": int(row["problem_number"]),
                "prompt": str(row["prompt"]),
                "rendered_prompt": str(row["rendered_prompt"]),
                "year": int(row["year"]),
            }
            if problem_id in problem_invariants and problem_invariants[problem_id] != invariant:
                raise ValueError(f"problem fields drift across draws: {problem_id}")
            problem_invariants[problem_id] = invariant
            metadata = {
                "base_trace_id": sample_id,
                "cap_hit": bool(row["cap_hit"]),
                "draw_index": draw_index,
                "finish_reason": row["finish_reason"],
                "is_correct": bool(row["is_correct"]),
                "problem_id": problem_id,
                "sample_id": sample_id,
                "scorer_decision_reason": str(row["decision_reason"]),
                "scorer_version": str(row["scorer_version"]),
                "source_dataset": contract["source_dataset"]["hf_repo"],
                "source_dataset_revision": contract["source_dataset"]["revision"],
                "source_output_tokens": int(row["output_tokens"]),
                "source_response_sha256": str(row["response_sha256"]),
                "trace_id": sample_id,
            }
            record = build_record(
                tokenizer,
                prompt=row["prompt"],
                rendered_prompt=row["rendered_prompt"],
                response=row["response"],
                metadata=metadata,
                max_sequence_length=int(contract["training"]["max_sequence_length"]),
            )
            rows.append(record)
    if len(rows) != 7680 or len(seen_samples) != 7680 or len(problem_invariants) != 480:
        raise ValueError("source shape is not 7,680 unique traces over 480 unique problems")
    if Counter(str(row["metadata"]["problem_id"]) for row in rows) != Counter(
        {problem_id: 16 for problem_id in problem_invariants}
    ):
        raise ValueError("source does not contain exactly 16 generations per problem")
    return rows, problem_invariants


def main() -> None:
    args = parse_args()
    contract = load_json(args.contract)
    if contract.get("family") != "qwen2_5_7b_aime_2009_2024_split_sft_6k":
        raise ValueError("experiment contract family drift")
    protected = [
        args.data_root / "datasets",
        args.data_root / "eval",
        args.data_root / "selection_and_tokenization_stats.json",
        args.data_root / "source_artifacts.json",
        args.data_root / "tokenizer_control_inventory.json",
    ]
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite preparation artifacts: {existing}")
    tokenizer = load_tokenizer(args.model, contract)
    rows, problems = load_source(args.source_dir, contract, tokenizer)
    seed = int(contract["selection"]["seed"])
    held_in, held_out = partition_problem_ids(problems, held_in_count=240, seed=seed)
    split_problem_ids = {"held_in": held_in, "held_out": held_out}
    args.data_root.mkdir(parents=True, exist_ok=True)
    (args.data_root / "datasets").mkdir()
    (args.data_root / "eval").mkdir()
    split_stats: dict[str, Any] = {}
    expansion_stats: dict[str, Any] = {}
    output_paths: dict[str, Path] = {}
    for split in SPLITS:
        problem_ids = split_problem_ids[split]
        split_rows = [row for row in rows if str(row["metadata"]["problem_id"]) in problem_ids]
        incorrect = [row for row in split_rows if not bool(row["metadata"]["is_correct"])]
        incorrect_problem_ids = {str(row["metadata"]["problem_id"]) for row in incorrect}
        correct_pool = [row for row in split_rows if bool(row["metadata"]["is_correct"])]
        selected_correct, correct_report = select_correct_control(
            correct_pool,
            target_rows=len(incorrect),
            target_problem_count=len(incorrect_problem_ids),
            seed=seed,
            split=split,
        )
        incorrect.sort(key=lambda row: str(row["metadata"]["base_trace_id"]))
        correct_problem_ids = {str(row["metadata"]["problem_id"]) for row in selected_correct}
        if len(selected_correct) != len(incorrect) or len(correct_problem_ids) != len(incorrect_problem_ids):
            raise AssertionError(f"{split} trace-count/problem-count control drift")
        split_stats[split] = {
            "correct_control": correct_report,
            "correct_problem_count": len(correct_problem_ids),
            "correct_problem_ids_sha256": ordered_values_sha256(sorted(correct_problem_ids)),
            "correct_rows": len(selected_correct),
            "held_problem_count": len(problem_ids),
            "held_problem_ids": sorted(problem_ids),
            "held_problem_ids_sha256": ordered_values_sha256(sorted(problem_ids)),
            "incorrect_problem_count": len(incorrect_problem_ids),
            "incorrect_problem_ids_sha256": ordered_values_sha256(sorted(incorrect_problem_ids)),
            "incorrect_rows": len(incorrect),
            "incorrect_trace_ids_sha256": ordered_values_sha256(
                str(row["metadata"]["base_trace_id"]) for row in incorrect
            ),
            "source_correct_pool_rows": len(correct_pool),
            "source_rows": len(split_rows),
        }
        for label, selected in (("correct", selected_correct), ("incorrect", incorrect)):
            variant = f"{split}_{label}_6000"
            expanded, expansion = expand_balanced(selected, variant=variant, seed=seed)
            path = args.data_root / "datasets" / f"{variant}.jsonl"
            atomic_jsonl(path, expanded)
            output_paths[variant] = path
            expansion_stats[variant] = {
                **expansion,
                "dataset_sha256": sha256_file(path),
                "length_stats": length_summary(expanded),
                "source_selection_length_stats": length_summary(selected),
            }
        eval_rows = []
        for problem_id in sorted(problem_ids):
            problem = problems[problem_id]
            eval_rows.append(
                {
                    "accepted_answer_integers": problem["accepted_answer_integers"],
                    "contest": problem["contest"],
                    "eval_index": problem["eval_index"],
                    "gold_answer_canonical": problem["gold_answer_canonical"],
                    "part": problem["part"],
                    "problem": problem["problem"],
                    "problem_id": problem_id,
                    "problem_number": problem["problem_number"],
                    "prompt": problem["prompt"],
                    "prompt_sha256": sha256_text(problem["prompt"]),
                    "rendered_prompt": problem["rendered_prompt"],
                    "rendered_prompt_sha256": sha256_text(problem["rendered_prompt"]),
                    "split": split,
                    "year": problem["year"],
                }
            )
        atomic_jsonl(args.data_root / "eval" / f"{split}_240.jsonl", eval_rows)
    if tuple(output_paths) != VARIANTS:
        raise AssertionError(f"variant output order drift: {tuple(output_paths)}")
    source_length_stats = length_summary(rows)
    expected_audit = contract["selection"]["realized_source_audit"]
    if source_length_stats["total_tokens"]["max"] != expected_audit["maximum_fully_rendered_tokens"]:
        raise ValueError("maximum fully rendered source length differs from the pinned audit")
    for split in SPLITS:
        observed = {
            "correct_problem_count": split_stats[split]["correct_problem_count"],
            "correct_problem_ids_sha256": split_stats[split]["correct_problem_ids_sha256"],
            "correct_rows": split_stats[split]["correct_rows"],
            "correct_trace_ids_sha256": split_stats[split]["correct_control"]["selected_trace_ids_sha256"],
            "held_problem_ids_sha256": split_stats[split]["held_problem_ids_sha256"],
            "incorrect_problem_count": split_stats[split]["incorrect_problem_count"],
            "incorrect_problem_ids_sha256": split_stats[split]["incorrect_problem_ids_sha256"],
            "incorrect_rows": split_stats[split]["incorrect_rows"],
            "incorrect_trace_ids_sha256": split_stats[split]["incorrect_trace_ids_sha256"],
        }
        if observed != expected_audit[split]:
            raise ValueError(f"{split} selection differs from the pinned source-backed audit")
        eval_hash = sha256_file(args.data_root / "eval" / f"{split}_240.jsonl")
        if eval_hash != expected_audit["eval_sha256"][split]:
            raise ValueError(f"{split} eval JSONL differs from the pinned source-backed audit")
    for variant in VARIANTS:
        if expansion_stats[variant]["dataset_sha256"] != expected_audit["dataset_sha256"][variant]:
            raise ValueError(f"{variant} dataset differs from the pinned source-backed audit")
    inventory_path = args.data_root / "tokenizer_control_inventory.json"
    atomic_json(inventory_path, tokenizer_control_inventory(tokenizer))
    stats_path = args.data_root / "selection_and_tokenization_stats.json"
    atomic_json(
        stats_path,
        {
            "artifact_schema_version": 1,
            "expansion": expansion_stats,
            "invariants": {
                "all_source_traces_retained_before_condition_selection": True,
                "correct_and_incorrect_problem_counts_match_within_each_split": True,
                "correct_and_incorrect_problem_identity_sets_may_differ": True,
                "correct_and_incorrect_source_trace_counts_match_within_each_split": True,
                "held_in_held_out_disjoint_exhaustive_240_each": True,
                "no_length_filter_or_truncation": True,
                "no_system_message": True,
                "prompt_and_response_bytes_preserved": True,
                "terminal_im_end_weight_one_newline_weight_zero": True,
            },
            "source": {
                "correct_rows": sum(bool(row["metadata"]["is_correct"]) for row in rows),
                "incorrect_rows": sum(not bool(row["metadata"]["is_correct"]) for row in rows),
                "length_stats": source_length_stats,
                "problem_count": len(problems),
                "rows": len(rows),
            },
            "splits": split_stats,
        },
    )
    atomic_json(
        args.data_root / "source_artifacts.json",
        {
            "artifact_schema_version": 1,
            "dataset": {
                "hf_repo": contract["source_dataset"]["hf_repo"],
                "revision": contract["source_dataset"]["revision"],
                "source_dir": str(args.source_dir.resolve()),
                "source_file_sha256": contract["source_dataset"]["files"],
            },
            "eval": {
                split: {
                    "path": str(args.data_root / "eval" / f"{split}_240.jsonl"),
                    "rows": 240,
                    "sha256": sha256_file(args.data_root / "eval" / f"{split}_240.jsonl"),
                }
                for split in SPLITS
            },
            "outputs": {
                variant: {"path": str(path), "sha256": sha256_file(path)} for variant, path in output_paths.items()
            },
            "selection_stats_sha256": sha256_file(stats_path),
            "tokenizer_inventory_sha256": sha256_file(inventory_path),
            "training_datasets_transferred": False,
        },
    )
    print(
        "AIME_SPLIT_6K_DATA_PREPARED "
        + " ".join(
            f"{split}_incorrect={split_stats[split]['incorrect_rows']} "
            f"{split}_problems={split_stats[split]['incorrect_problem_count']}"
            for split in SPLITS
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
