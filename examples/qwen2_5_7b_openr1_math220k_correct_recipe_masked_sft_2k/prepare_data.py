#!/usr/bin/env python3
"""Build one frozen 1K-correct plus 1K-wrong Qwen2.5-7B trace set."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    atomic_json,
    atomic_jsonl,
    build_training_record,
    normalize_token_ids,
    sha256_file,
    token_sha256,
    tokenizer_control_inventory,
    trace_set_sha256,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    correctness_at,
    problem_id,
    trace_structure_ok,
    usable_text,
    user_content as margin_user_content,
    value_at,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-selected", type=Path, required=True)
    parser.add_argument("--source-pretokenized", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--unmasked-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correct-rows", type=int, default=1000)
    parser.add_argument("--wrong-rows", type=int, default=1000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_selected(row: dict[str, Any], tokenizer: Any, is_correct: bool) -> dict[str, Any]:
    trace = str(row["assistant_trace"])
    assistant_ids = normalize_token_ids(tokenizer.encode(trace, add_special_tokens=False))
    result = {
        "answer": str(row["answer"]),
        "assistant_token_count": len(assistant_ids),
        "assistant_token_sha256": token_sha256(assistant_ids),
        "assistant_trace": trace,
        "correctness_source": str(row["correctness_source"]),
        "generation_index": int(row["generation_index"]),
        "is_correct": is_correct,
        "problem": str(row["problem"]),
        "problem_id": str(row["problem_id"]),
        "source": row.get("source"),
        "source_row_index": int(row["source_row_index"]),
        "trace_id": str(row["trace_id"]),
    }
    return result


def score_sequence_fits(
    tokenizer: Any,
    problem: str,
    answer: str,
    assistant_ids: list[int],
    max_sequence_length: int,
) -> bool:
    for conditioned in (False, True):
        prompt = normalize_token_ids(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": margin_user_content(problem, answer if conditioned else None),
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if len(prompt) + len(assistant_ids) > max_sequence_length:
            return False
    return True


def eligible_wrong(
    source: dict[str, Any],
    row_index: int,
    tokenizer: Any,
    max_sequence_length: int,
    trace_token_cap: int,
    seed: int,
    counts: Counter[str],
) -> dict[str, Any] | None:
    generations = source.get("generations")
    problem = usable_text(source.get("problem"))
    answer = usable_text(source.get("answer"))
    if not isinstance(generations, (list, tuple)) or problem is None or answer is None:
        counts["row_missing_required_fields"] += 1
        return None
    indices = list(range(len(generations)))
    random.Random(f"{seed}:{row_index}:wrong-candidate").shuffle(indices)
    for generation_index in indices:
        correct, correctness_source = correctness_at(source, generation_index)
        if correct is not False:
            continue
        counts["wrong_candidates_seen"] += 1
        if value_at(source.get("is_reasoning_complete"), generation_index) is not True:
            counts["wrong_incomplete_source_flag"] += 1
            continue
        trace = usable_text(generations[generation_index])
        if trace is None:
            counts["wrong_empty"] += 1
            continue
        structure_ok, reason = trace_structure_ok(trace)
        if not structure_ok:
            counts[f"wrong_rejected_{reason}"] += 1
            continue
        assistant_ids = normalize_token_ids(tokenizer.encode(trace, add_special_tokens=False))
        if len(assistant_ids) > trace_token_cap:
            counts["wrong_trace_too_long"] += 1
            continue
        selected = {
            "answer": answer,
            "assistant_token_count": len(assistant_ids),
            "assistant_token_sha256": token_sha256(assistant_ids),
            "assistant_trace": trace,
            "correctness_source": str(correctness_source),
            "generation_index": generation_index,
            "is_correct": False,
            "problem": problem,
            "problem_id": problem_id(source, row_index),
            "source": source.get("source"),
            "source_row_index": row_index,
            "trace_id": (f"{problem_id(source, row_index)}__row_{row_index:07d}" f"__gen_{generation_index:03d}"),
        }
        try:
            build_training_record(tokenizer, selected, None, max_sequence_length, {})
        except ValueError:
            counts["wrong_training_sequence_rejected"] += 1
            continue
        if not score_sequence_fits(tokenizer, problem, answer, assistant_ids, max_sequence_length):
            counts["wrong_margin_sequence_too_long"] += 1
            continue
        return selected
    counts["problem_no_eligible_wrong_trace"] += 1
    return None


def select_wrong_rows(
    dataset: Any,
    excluded_problem_ids: set[str],
    tokenizer: Any,
    target: int,
    max_sequence_length: int,
    trace_token_cap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    counts: Counter[str] = Counter()
    reservoir: list[dict[str, Any]] = []
    eligible_seen = 0
    seen_problem_ids: set[str] = set()
    rng = random.Random(f"{seed}:wrong:reservoir")
    for row_index, source in enumerate(dataset):
        counts["source_rows_seen"] += 1
        pid = problem_id(source, row_index)
        if pid in seen_problem_ids:
            counts["duplicate_problem_id"] += 1
            continue
        seen_problem_ids.add(pid)
        if pid in excluded_problem_ids:
            counts["excluded_correct_only_problem"] += 1
            continue
        selected = eligible_wrong(
            source,
            row_index,
            tokenizer,
            max_sequence_length,
            trace_token_cap,
            seed,
            counts,
        )
        if selected is None:
            continue
        eligible_seen += 1
        if len(reservoir) < target:
            reservoir.append(selected)
        else:
            slot = rng.randrange(eligible_seen)
            if slot < target:
                reservoir[slot] = selected
        if counts["source_rows_seen"] % 5000 == 0:
            print(
                f"WRONG_SELECTION_PROGRESS rows={counts['source_rows_seen']} "
                f"eligible={eligible_seen} reservoir={len(reservoir)}",
                flush=True,
            )
    if len(reservoir) != target:
        raise RuntimeError(f"wanted {target} eligible wrong rows, found {len(reservoir)}")
    random.Random(f"{seed}:wrong:final-order").shuffle(reservoir)
    for wrong_index, row in enumerate(reservoir):
        row["wrong_index"] = wrong_index
    return reservoir, counts, eligible_seen


def assert_correct_record_matches_source(record: dict[str, Any], source: dict[str, Any]) -> None:
    trace_id = record["metadata"]["trace_id"]
    if record["input_ids"] != source["input_ids"]:
        raise ValueError(f"correct input IDs drift from source: {trace_id}")
    for key in (
        "assistant_token_count",
        "assistant_token_sha256",
        "loss_weights",
        "problem_id",
        "response_length",
        "trace_id",
    ):
        if record["metadata"][key] != source["metadata"][key]:
            raise ValueError(f"correct metadata drift for {key}: {trace_id}")


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    selection = contract["selection"]
    for path in (
        args.selected_output,
        args.unmasked_output,
        args.stats_output,
        args.inventory_output,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if sha256_file(args.source_selected) != selection["source_correct_only_selected_sha256"]:
        raise ValueError("frozen correct-only selection hash drift")
    if sha256_file(args.source_pretokenized) != selection["source_correct_only_dataset_sha256"]:
        raise ValueError("frozen correct-only token dataset hash drift")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    frozen = read_jsonl(args.source_selected)
    frozen_records = read_jsonl(args.source_pretokenized)
    if len(frozen) != 2000 or len(frozen_records) != 2000:
        raise ValueError("frozen correct-only source is not exactly 2,000 rows")
    frozen_problem_ids = {str(row["problem_id"]) for row in frozen}
    if len(frozen_problem_ids) != 2000:
        raise ValueError("frozen correct-only problem IDs are not unique")
    correct = [normalized_selected(row, tokenizer, True) for row in frozen[: args.correct_rows]]
    source_by_trace = {str(row["metadata"]["trace_id"]): row for row in frozen_records}
    if len(source_by_trace) != len(frozen_records):
        raise ValueError("duplicate source pretokenized trace ID")

    dataset_spec = selection["source_dataset"]
    dataset = load_dataset(
        dataset_spec["hf_repo"],
        dataset_spec["config"],
        split=dataset_spec["split"],
        revision=dataset_spec["revision"],
        cache_dir=args.cache_dir,
    )
    wrong, skip_counts, eligible_seen = select_wrong_rows(
        dataset,
        frozen_problem_ids,
        tokenizer,
        args.wrong_rows,
        int(contract["training"]["max_sequence_length"]),
        int(contract["training"]["trace_token_cap"]),
        args.seed,
    )
    mixed = correct + wrong
    if len({row["trace_id"] for row in mixed}) != len(mixed):
        raise ValueError("mixed trace IDs are not unique")
    if len({row["problem_id"] for row in mixed}) != len(mixed):
        raise ValueError("mixed problem IDs are not unique")

    records: list[dict[str, Any]] = []
    for row in mixed:
        record = build_training_record(
            tokenizer,
            row,
            None,
            int(contract["training"]["max_sequence_length"]),
            {},
        )
        if row["is_correct"]:
            assert_correct_record_matches_source(record, source_by_trace[row["trace_id"]])
        records.append(record)
    atomic_jsonl(args.selected_output, mixed)
    atomic_jsonl(args.unmasked_output, records)
    atomic_json(args.inventory_output, tokenizer_control_inventory(tokenizer))
    atomic_json(
        args.stats_output,
        {
            "artifact_schema_version": 1,
            "counts": {
                "correct": len(correct),
                "frozen_correct_only": len(frozen),
                "mixed": len(mixed),
                "wrong": len(wrong),
                "wrong_eligible_seen": eligible_seen,
            },
            "invariants": {
                "correct_input_ids_match_source": True,
                "correct_rows_are_first_frozen_1000": True,
                "correct_wrong_problem_ids_disjoint": True,
                "same_order_required_for_every_variant": True,
                "source_correct_only_immutable": True,
            },
            "mask_seed": args.seed,
            "ordered_mixed_trace_ids_sha256": trace_set_sha256([str(row["trace_id"]) for row in mixed]),
            "ordered_correct_trace_ids_sha256": trace_set_sha256([str(row["trace_id"]) for row in correct]),
            "ordered_wrong_trace_ids_sha256": trace_set_sha256([str(row["trace_id"]) for row in wrong]),
            "outputs": {
                "selected_sha256": sha256_file(args.selected_output),
                "unmasked_sha256": sha256_file(args.unmasked_output),
            },
            "selection_policy": {
                "correct": "first 1000 of frozen seed-42 correct-only order",
                "mixed_order": "correct then wrong",
                "wrong": "seed-42 reservoir over unique eligible problems excluding all frozen correct-only problem IDs",
            },
            "source": {
                "correct_only_dataset_sha256": sha256_file(args.source_pretokenized),
                "correct_only_selected_sha256": sha256_file(args.source_selected),
                "dataset_revision": dataset_spec["revision"],
            },
            "wrong_skip_counts": dict(sorted(skip_counts.items())),
        },
    )
    print(
        f"CORRECT_RECIPE_DATA_COMPLETE correct={len(correct)} wrong={len(wrong)} " f"eligible_wrong={eligible_seen}",
        flush=True,
    )


if __name__ == "__main__":
    main()
