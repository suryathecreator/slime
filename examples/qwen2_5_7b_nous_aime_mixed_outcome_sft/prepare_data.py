#!/usr/bin/env python3
"""Rescore Nous AIME generations and build four balanced 3K SFT datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.aime_scorer import (
    SCORER_VERSION,
    score_response,
)
from examples.qwen2_5_7b_nous_aime_mixed_outcome_sft.data_utils import (
    VARIANTS,
    YEARS,
    atomic_json,
    atomic_jsonl,
    build_record,
    conditionally_uniform_subset,
    expand_balanced,
    length_summary,
    ordered_values_sha256,
    select_conditioned_problem_subset,
    sha256_file,
    sha256_text,
    source_trace_id,
    tokenizer_control_inventory,
)

REQUIRED_FIELDS = {
    "benchmark",
    "doc_id",
    "gold_choice",
    "messages",
    "num_samples",
    "query",
    "response_id",
    "sample_id",
    "task_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--aime24-source", type=Path, required=True)
    parser.add_argument("--aime25-source", type=Path, required=True)
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
            raise ValueError(f"tokenizer source hash drift: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    expected_controls = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
    }
    inventory = tokenizer_control_inventory(tokenizer)
    for token, token_id in expected_controls.items():
        if inventory["added_vocabulary"].get(token) != token_id:
            raise ValueError(f"tokenizer control drift for {token}")
    return tokenizer


def message_content(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Nous messages must be a three-message list")
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"unexpected Nous message roles: {roles}")
    if messages[0].get("content") != "/think":
        raise ValueError("Nous source system message is not exact /think")
    for message in messages:
        if message["role"] == role:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Nous {role} content must be nonempty text")
            return content
    raise AssertionError(f"missing role after role validation: {role}")


def validate_row(row: dict[str, Any], year: str) -> tuple[str, int, int, str, str]:
    if set(row) != REQUIRED_FIELDS:
        raise ValueError(
            f"Nous schema drift: missing={sorted(REQUIRED_FIELDS - set(row))} "
            f"extra={sorted(set(row) - REQUIRED_FIELDS)}"
        )
    if row["benchmark"] != year:
        raise ValueError(f"benchmark/year mismatch: {row['benchmark']} != {year}")
    doc_id = str(row["doc_id"])
    if not doc_id.isdigit() or not 0 <= int(doc_id) < 30:
        raise ValueError(f"invalid {year} doc_id: {doc_id}")
    response_id = int(row["response_id"])
    if not 0 <= response_id < 64 or int(row["num_samples"]) != 64:
        raise ValueError(f"invalid {year}/{doc_id} response metadata")
    gold_text = str(row["gold_choice"])
    if not gold_text.isdigit() or not 0 <= int(gold_text) <= 999:
        raise ValueError(f"invalid AIME gold: {gold_text!r}")
    query = row["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Nous query must be nonempty text")
    user = message_content(row, "user")
    assistant = message_content(row, "assistant")
    if query != user:
        raise ValueError(f"query/user bytes differ for {year}/{doc_id}/{response_id}")
    return doc_id, response_id, int(gold_text), query, assistant


def score_year(
    source: Path,
    *,
    year: str,
    tokenizer: Any,
    max_sequence_length: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(source).to_pylist()
    if len(rows) != 1920:
        raise ValueError(f"{year} source must contain 1920 rows, found {len(rows)}")
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_invariants: dict[str, tuple[str, int]] = {}
    seen: set[tuple[str, int]] = set()
    excluded_over_cap: list[str] = []
    decision_counts: Counter[str] = Counter()
    eligible_lengths: list[int] = []
    for row in rows:
        doc_id, response_id, gold, query, assistant = validate_row(row, year)
        location = (doc_id, response_id)
        if location in seen:
            raise ValueError(f"duplicate source trajectory: {year}/{doc_id}/{response_id}")
        seen.add(location)
        invariant = (query, gold)
        if doc_id in eval_invariants and eval_invariants[doc_id] != invariant:
            raise ValueError(f"query/gold drift within {year}/{doc_id}")
        eval_invariants[doc_id] = invariant
        base_trace_id = source_trace_id(year, doc_id, response_id)
        metadata = {
            "base_trace_id": base_trace_id,
            "doc_id": doc_id,
            "gold_answer": gold,
            "problem_id": f"{year}:{doc_id}",
            "response_id": response_id,
            "source_benchmark": year,
            "source_dataset": "NousResearch/eval-Qwen3-235B-A22B-reasoning",
            "source_sample_id": str(row["sample_id"]),
            "trace_id": base_trace_id,
            "year": int(year[-2:]) + 2000,
        }
        try:
            record = build_record(
                tokenizer,
                query=query,
                assistant=assistant,
                metadata=metadata,
                max_sequence_length=max_sequence_length,
            )
        except ValueError as error:
            if str(error).startswith("rendered sequence has "):
                excluded_over_cap.append(base_trace_id)
                continue
            raise
        if not assistant.startswith("<think>\n") or "</think>" not in assistant:
            raise ValueError(f"eligible assistant think structure drift for {year}/{doc_id}/{response_id}")
        decision = score_response(
            assistant,
            gold,
            finish_reason="source_dataset",
            stop_token_id=None,
            cap_hit=False,
            problem_id=f"{year}:{doc_id}",
        )
        record["metadata"].update(
            {
                "correctness_source": SCORER_VERSION,
                "is_correct": bool(decision["is_correct"]),
                "scorer_decision_reason": decision["official_decision_reason"],
                "scorer_parsed_integer": decision["parsed_integer"],
            }
        )
        decision_counts[decision["official_decision_reason"]] += 1
        eligible_lengths.append(len(record["input_ids"]))
        by_doc[doc_id].append(record)
    if set(eval_invariants) != {str(index) for index in range(30)}:
        raise ValueError(f"{year} does not contain exactly doc_id 0..29")
    if any(len([row for row in rows if str(row["doc_id"]) == doc_id]) != 64 for doc_id in eval_invariants):
        raise ValueError(f"{year} does not contain 64 source rows per problem")
    eval_rows = [
        {
            "answer": gold,
            "doc_id": doc_id,
            "global_eval_index": int(year[-2:] == "25") * 30 + int(doc_id),
            "problem_id": f"{year}:{doc_id}",
            "query": query,
            "query_sha256": sha256_text(query),
            "year": int(year[-2:]) + 2000,
        }
        for doc_id, (query, gold) in sorted(eval_invariants.items(), key=lambda item: int(item[0]))
    ]
    report = {
        "decision_counts": dict(sorted(decision_counts.items())),
        "eligible_rows": sum(len(records) for records in by_doc.values()),
        "longest_eligible_total_tokens": max(eligible_lengths),
        "over_32768_base_trace_ids_sha256": ordered_values_sha256(sorted(excluded_over_cap)),
        "over_32768_rows": len(excluded_over_cap),
        "source_rows": len(rows),
    }
    return by_doc, report, eval_rows


def select_year(
    by_doc: dict[str, list[dict[str, Any]]],
    *,
    year: str,
    seed: int,
    selected_doc_ids: set[str],
    full_mixed_doc_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not selected_doc_ids or not selected_doc_ids <= full_mixed_doc_ids:
        raise ValueError(f"{year} selected problems are not a nonempty subset of the mixed pool")
    incorrect = [
        row
        for doc_id in sorted(selected_doc_ids, key=int)
        for row in by_doc[doc_id]
        if not bool(row["metadata"]["is_correct"])
    ]
    correct_pool = [
        row
        for doc_id in sorted(selected_doc_ids, key=int)
        for row in by_doc[doc_id]
        if bool(row["metadata"]["is_correct"])
    ]
    selected_correct, attempt = conditionally_uniform_subset(
        correct_pool,
        count=len(incorrect),
        required_doc_ids=selected_doc_ids,
        seed=seed,
        year=year,
    )
    correct_docs = {str(row["metadata"]["doc_id"]) for row in selected_correct}
    incorrect_docs = {str(row["metadata"]["doc_id"]) for row in incorrect}
    if correct_docs != incorrect_docs or correct_docs != selected_doc_ids:
        raise AssertionError(f"{year} correct/incorrect mixed-problem coverage drift")
    selected_correct.sort(key=lambda row: str(row["metadata"]["base_trace_id"]))
    incorrect.sort(key=lambda row: str(row["metadata"]["base_trace_id"]))
    per_problem = {
        doc_id: {
            "eligible_correct": sum(bool(row["metadata"]["is_correct"]) for row in by_doc[doc_id]),
            "eligible_incorrect": sum(not bool(row["metadata"]["is_correct"]) for row in by_doc[doc_id]),
            "selected_correct": sum(str(row["metadata"]["doc_id"]) == doc_id for row in selected_correct),
            "selected_incorrect": sum(str(row["metadata"]["doc_id"]) == doc_id for row in incorrect),
        }
        for doc_id in sorted(selected_doc_ids, key=int)
    }
    report = {
        "correct_pool_rows": len(correct_pool),
        "correct_selection_attempt": attempt,
        "full_mixed_doc_ids": sorted(full_mixed_doc_ids, key=int),
        "mixed_doc_ids": sorted(selected_doc_ids, key=int),
        "per_problem": per_problem,
        "selected_correct_rows": len(selected_correct),
        "selected_correct_trace_ids_sha256": ordered_values_sha256(
            [str(row["metadata"]["base_trace_id"]) for row in selected_correct]
        ),
        "selected_incorrect_rows": len(incorrect),
        "selected_incorrect_trace_ids_sha256": ordered_values_sha256(
            [str(row["metadata"]["base_trace_id"]) for row in incorrect]
        ),
    }
    return selected_correct, incorrect, report


def main() -> None:
    args = parse_args()
    contract = load_json(args.contract)
    if contract.get("family") != "qwen2_5_7b_nous_aime_mixed_outcome_sft":
        raise ValueError("experiment contract family drift")
    protected = (
        args.data_root / "datasets",
        args.data_root / "eval",
        args.data_root / "selection_and_tokenization_stats.json",
        args.data_root / "source_artifacts.json",
        args.data_root / "tokenizer_control_inventory.json",
    )
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite preparation artifacts: {existing}")
    dataset = contract["selection"]["dataset"]
    source_paths = {"aime24": args.aime24_source, "aime25": args.aime25_source}
    for year, source in source_paths.items():
        expected = dataset[f"{year}_sha256"]
        if sha256_file(source) != expected:
            raise ValueError(f"pinned source hash drift: {source}")
    tokenizer = load_tokenizer(args.model, contract)
    args.data_root.mkdir(parents=True, exist_ok=True)
    (args.data_root / "datasets").mkdir()
    (args.data_root / "eval").mkdir()
    max_length = int(contract["training"]["max_sequence_length"])
    seed = int(contract["selection"]["seed"])
    scorer_path = (
        Path(__file__).resolve().parent.parent / "qwen2_5_7b_aime_generalization_incorrect_sft/eval/aime_scorer.py"
    )
    all_stats: dict[str, Any] = {}
    expansion_stats: dict[str, Any] = {}
    output_paths: dict[str, Path] = {}
    scored: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        by_doc, source_report, eval_rows = score_year(
            source_paths[year],
            year=year,
            tokenizer=tokenizer,
            max_sequence_length=max_length,
        )
        full_mixed_doc_ids = {
            doc_id
            for doc_id, records in by_doc.items()
            if any(bool(row["metadata"]["is_correct"]) for row in records)
            and any(not bool(row["metadata"]["is_correct"]) for row in records)
        }
        if not full_mixed_doc_ids:
            raise ValueError(f"{year} has no post-rescore mixed-outcome problems")
        scored[year] = {
            "by_doc": by_doc,
            "eval_rows": eval_rows,
            "full_mixed_doc_ids": full_mixed_doc_ids,
            "source_report": source_report,
        }

    subset_contract = contract["selection"]["conditioned_problem_subset"]
    problem_count = int(subset_contract["problem_count"])
    epsilon = int(subset_contract["incorrect_trace_epsilon"])
    aime24_mixed = set(scored["aime24"]["full_mixed_doc_ids"])
    if len(aime24_mixed) != problem_count:
        raise ValueError(f"aime24 must realize exactly {problem_count} mixed problems, found {len(aime24_mixed)}")
    aime24_incorrect_rows = sum(
        not bool(row["metadata"]["is_correct"])
        for doc_id in aime24_mixed
        for row in scored["aime24"]["by_doc"][doc_id]
    )
    selected_doc_ids: dict[str, set[str]] = {"aime24": aime24_mixed}
    subset_reports: dict[str, dict[str, Any]] = {
        "aime24": {
            "algorithm": "all_post_rescore_mixed_problems",
            "epsilon": epsilon,
            "full_mixed_doc_ids": sorted(aime24_mixed, key=int),
            "full_mixed_incorrect_counts": {
                doc_id: sum(not bool(row["metadata"]["is_correct"]) for row in scored["aime24"]["by_doc"][doc_id])
                for doc_id in sorted(aime24_mixed, key=int)
            },
            "full_mixed_problem_count": len(aime24_mixed),
            "problem_count": problem_count,
            "selected_delta": 0,
            "selected_doc_ids": sorted(aime24_mixed, key=int),
            "selected_incorrect_rows": aime24_incorrect_rows,
            "target_incorrect_rows": aime24_incorrect_rows,
        }
    }
    aime25_mixed = set(scored["aime25"]["full_mixed_doc_ids"])
    aime25_incorrect_counts = {
        doc_id: sum(not bool(row["metadata"]["is_correct"]) for row in scored["aime25"]["by_doc"][doc_id])
        for doc_id in aime25_mixed
    }
    selected_doc_ids["aime25"], subset_reports["aime25"] = select_conditioned_problem_subset(
        aime25_incorrect_counts,
        problem_count=problem_count,
        target_incorrect_rows=aime24_incorrect_rows,
        epsilon=epsilon,
        seed=int(subset_contract["seed"]),
        year="aime25",
    )

    for year in YEARS:
        by_doc = scored[year]["by_doc"]
        eval_rows = scored[year]["eval_rows"]
        full_mixed_doc_ids = set(scored[year]["full_mixed_doc_ids"])
        source_report = scored[year]["source_report"]
        correct, incorrect, selection_report = select_year(
            by_doc,
            year=year,
            seed=seed,
            selected_doc_ids=selected_doc_ids[year],
            full_mixed_doc_ids=full_mixed_doc_ids,
        )
        held_in = set(selection_report["mixed_doc_ids"])
        for row in eval_rows:
            row["split"] = "held_in" if row["doc_id"] in held_in else "held_out"
        eval_path = args.data_root / "eval" / f"{year}_30.jsonl"
        atomic_jsonl(eval_path, eval_rows)
        all_stats[year] = {
            "expected_post_rescore_diagnostic": contract["selection"]["expected_post_rescore_diagnostic"][year],
            "observed_differs_from_expected": (
                selection_report["mixed_doc_ids"]
                != contract["selection"]["expected_post_rescore_diagnostic"][year]["selected_mixed_doc_ids"]
                or selection_report["full_mixed_doc_ids"]
                != contract["selection"]["expected_post_rescore_diagnostic"][year]["full_mixed_doc_ids"]
                or selection_report["selected_incorrect_rows"]
                != contract["selection"]["expected_post_rescore_diagnostic"][year]["incorrect_rows"]
            ),
            "problem_subset_selection": subset_reports[year],
            "selection": selection_report,
            "source_audit": source_report,
        }
        for correctness, selected in (("correct", correct), ("incorrect", incorrect)):
            variant = f"{year}_{correctness}_3000"
            expanded, expansion = expand_balanced(selected, variant=variant, seed=seed)
            path = args.data_root / "datasets" / f"{variant}.jsonl"
            atomic_jsonl(path, expanded)
            output_paths[variant] = path
            expansion_stats[variant] = {
                **expansion,
                "dataset_sha256": sha256_file(path),
                "length_stats": length_summary(expanded),
            }
    if tuple(output_paths) != VARIANTS:
        raise AssertionError(f"variant output order drift: {tuple(output_paths)}")
    inventory_path = args.data_root / "tokenizer_control_inventory.json"
    atomic_json(inventory_path, tokenizer_control_inventory(tokenizer))
    stats_path = args.data_root / "selection_and_tokenization_stats.json"
    atomic_json(
        stats_path,
        {
            "artifact_schema_version": 1,
            "expansion": expansion_stats,
            "invariants": {
                "all_incorrect_from_selected_rescored_mixed_problems": True,
                "correct_and_incorrect_exact_same_problem_set_per_year": True,
                "correct_and_incorrect_same_source_trace_count_per_year": True,
                "equal_selected_problem_count_across_all_variants": True,
                "equal_source_trace_count_across_all_variants": True,
                "no_synthetic_endoftext": True,
                "query_equals_source_user_message_byte_for_byte": True,
                "runtime_source_exposure_max_difference": 1,
                "source_system_think_omitted": True,
                "terminal_im_end_weight_one_newline_weight_zero": True,
                "training_and_eval_query_bytes_identical": True,
            },
            "scorer": {
                "path": str(scorer_path),
                "sha256": sha256_file(scorer_path),
                "version": SCORER_VERSION,
            },
            "years": all_stats,
        },
    )
    atomic_json(
        args.data_root / "source_artifacts.json",
        {
            "artifact_schema_version": 1,
            "datasets": {
                year: {"path": str(path), "sha256": sha256_file(path)} for year, path in source_paths.items()
            },
            "eval": {
                year: {
                    "path": str(args.data_root / "eval" / f"{year}_30.jsonl"),
                    "sha256": sha256_file(args.data_root / "eval" / f"{year}_30.jsonl"),
                }
                for year in YEARS
            },
            "outputs": {
                variant: {"path": str(path), "sha256": sha256_file(path)} for variant, path in output_paths.items()
            },
            "scorer_sha256": sha256_file(scorer_path),
            "selection_stats_sha256": sha256_file(stats_path),
            "tokenizer_inventory_sha256": sha256_file(inventory_path),
            "training_datasets_transferred": False,
        },
    )
    print(
        "NOUS_AIME_DATA_PREPARED "
        + " ".join(
            f"{year}_mixed={len(all_stats[year]['selection']['mixed_doc_ids'])} "
            f"{year}_traces={all_stats[year]['selection']['selected_incorrect_rows']}"
            for year in YEARS
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
