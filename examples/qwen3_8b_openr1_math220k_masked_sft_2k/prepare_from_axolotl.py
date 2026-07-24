#!/usr/bin/env python3
"""Retokenize the canonical Axolotl unpaired 2K traces for Slime full SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.qwen3_8b_openr1_math220k_masked_sft_2k.data_utils import (
    protected_think_positions,
    source_location,
    trace_set_sha256,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    atomic_json,
    atomic_jsonl,
    build_training_record,
    sha256_file,
    token_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-correct-only", required=True)
    parser.add_argument("--source-mixed-unmasked", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--selected-output", required=True)
    parser.add_argument("--correct-output", required=True)
    parser.add_argument("--mixed-output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_trace_structure(trace: str) -> tuple[bool, str]:
    """Accept the frozen Axolotl traces without silently filtering them."""
    if any(
        token in trace
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
    ):
        return False, "embedded_chat_control"
    open_count = trace.count("<think>")
    close_count = trace.count("</think>")
    if open_count != 1:
        return False, "opening_think_tag_count"
    start = trace.index("<think>") + len("<think>")
    if close_count == 0:
        return (
            (True, "missing_close_preserved")
            if trace[start:].strip()
            else (False, "empty_after_open")
        )
    closes: list[int] = []
    cursor = 0
    while True:
        position = trace.find("</think>", cursor)
        if position < 0:
            break
        closes.append(position)
        cursor = position + len("</think>")
    if any(position < start for position in closes):
        return False, "think_tag_order"
    if not trace[closes[-1] + len("</think>") :].strip():
        return False, "empty_post_think"
    return (
        (True, "ok")
        if close_count == 1
        else (True, "duplicate_close_preserved")
    )


def normalize_source(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError(f"unexpected source conversation: {row.get('id')}")
    metadata = row.get("metadata", {})
    trace_id = str(metadata.get("trace_id") or str(row["id"]).split("/", 1)[-1])
    source_row, generation = source_location(trace_id)
    user_text = str(messages[0]["content"])
    marker = "\n\nProblem:\n"
    if marker not in user_text:
        raise ValueError(f"source prompt has no problem marker: {trace_id}")
    problem = user_text.split(marker, 1)[1]
    assistant = str(messages[1]["content"])
    structure_ok, structure_reason = canonical_trace_structure(assistant)
    if not structure_ok:
        raise ValueError(f"invalid assistant structure for {trace_id}: {structure_reason}")
    assistant_ids = [
        int(value) for value in tokenizer.encode(assistant, add_special_tokens=False)
    ]
    if not assistant_ids:
        raise ValueError(f"empty assistant trace: {trace_id}")
    return {
        "answer": str(metadata["correct_answer"]),
        "assistant_token_count": len(assistant_ids),
        "assistant_token_sha256": token_sha256(assistant_ids),
        "assistant_trace": assistant,
        "closing_think_tag_count": assistant.count("</think>"),
        "correctness_source": "axolotl_metadata.is_correct",
        "generation_index": generation,
        "is_correct": bool(metadata["is_correct"]),
        "problem": problem,
        "problem_id": str(metadata["problem_id"]),
        "source_row_index": source_row,
        "trace_id": trace_id,
    }


def add_terminal_audit(
    record: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    metadata = record["metadata"]
    assistant_count = int(metadata["assistant_token_count"])
    response_length = int(metadata["response_length"])
    response_ids = [int(value) for value in record["input_ids"][-response_length:]]
    assistant_ids = response_ids[:assistant_count]
    positions = protected_think_positions(tokenizer, assistant_ids)
    metadata["protected_assistant_positions"] = positions
    weights = [float(value) for value in metadata["loss_weights"]]
    if any(weights[index] != 1.0 for index in positions):
        raise ValueError(f"think tag is not supervised: {metadata['trace_id']}")
    return record


def main() -> None:
    args = parse_args()
    destinations = (
        args.selected_output,
        args.correct_output,
        args.mixed_output,
        args.stats_output,
    )
    for destination in destinations:
        if Path(destination).exists():
            raise FileExistsError(f"refusing to overwrite {destination}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    correct = [
        normalize_source(row, tokenizer)
        for row in load_jsonl(args.source_correct_only)
    ]
    mixed = [
        normalize_source(row, tokenizer)
        for row in load_jsonl(args.source_mixed_unmasked)
    ]
    if len(correct) != 2000 or len(mixed) != 2000:
        raise ValueError(f"canonical row count drift: correct={len(correct)} mixed={len(mixed)}")
    if any(not row["is_correct"] for row in correct):
        raise ValueError("correct-only source contains a wrong trace")
    mixed_correct = [row for row in mixed if row["is_correct"]]
    mixed_wrong = [row for row in mixed if not row["is_correct"]]
    if len(mixed_correct) != 1000 or len(mixed_wrong) != 1000:
        raise ValueError("mixed source is not 1000 correct plus 1000 wrong")
    if {row["problem_id"] for row in mixed_correct} & {
        row["problem_id"] for row in mixed_wrong
    }:
        raise ValueError("mixed source is paired; expected disjoint unpaired prompts")
    correct_by_trace = {row["trace_id"]: row for row in correct}
    if len(correct_by_trace) != len(correct):
        raise ValueError("duplicate correct-only trace")
    for row in mixed_correct:
        canonical = correct_by_trace.get(row["trace_id"])
        if canonical is None or canonical != row:
            raise ValueError(f"mixed correct trace is not an exact correct-only subset: {row['trace_id']}")

    for wrong_index, row in enumerate(mixed_wrong):
        row["wrong_index"] = wrong_index
    selected_union = correct + mixed_wrong
    if len({row["trace_id"] for row in selected_union}) != 3000:
        raise ValueError("selected union contains duplicate trace IDs")

    correct_records = (
        add_terminal_audit(
            build_training_record(tokenizer, row, None, args.max_sequence_length),
            tokenizer,
        )
        for row in correct
    )
    mixed_records = (
        add_terminal_audit(
            build_training_record(tokenizer, row, None, args.max_sequence_length),
            tokenizer,
        )
        for row in mixed
    )
    atomic_jsonl(args.selected_output, selected_union)
    atomic_jsonl(args.correct_output, correct_records)
    atomic_jsonl(args.mixed_output, mixed_records)
    stats = {
        "selection_mode": "canonical_axolotl_unpaired_seed_42",
        "training_pipeline": "slime_speedy_full_parameter_sft",
        "counts": {
            "correct_only": len(correct),
            "mixed_correct": len(mixed_correct),
            "mixed_wrong": len(mixed_wrong),
            "selected_union": len(selected_union),
        },
        "trace_sets": {
            "correct_only_ordered_sha256": trace_set_sha256(
                [row["trace_id"] for row in correct]
            ),
            "mixed_ordered_sha256": trace_set_sha256(
                [row["trace_id"] for row in mixed]
            ),
            "mixed_correct_ordered_sha256": trace_set_sha256(
                [row["trace_id"] for row in mixed_correct]
            ),
            "mixed_wrong_ordered_sha256": trace_set_sha256(
                [row["trace_id"] for row in mixed_wrong]
            ),
        },
        "source_artifacts": {
            "correct_only_sha256": sha256_file(args.source_correct_only),
            "correct_wrong_unmasked_sha256": sha256_file(
                args.source_mixed_unmasked
            ),
        },
        "source_structure": {
            "correct_only_missing_close": sum(
                row["closing_think_tag_count"] == 0 for row in correct
            ),
            "correct_only_duplicate_close": sum(
                row["closing_think_tag_count"] > 1 for row in correct
            ),
            "mixed_missing_close": sum(
                row["closing_think_tag_count"] == 0 for row in mixed
            ),
            "mixed_duplicate_close": sum(
                row["closing_think_tag_count"] > 1 for row in mixed
            ),
            "policy": "preserve canonical assistant bytes; supervise every literal think-tag token present",
        },
        "output_artifacts": {
            "selected_union_sha256": sha256_file(args.selected_output),
            "correct_only_sha256": sha256_file(args.correct_output),
            "mixed_unmasked_sha256": sha256_file(args.mixed_output),
        },
        "prompt_policy": "Slime OpenR1 user_content with Qwen3 thinking generation boundary",
        "terminal_policy": "every present literal think tag and im_end weight 1; no synthetic endoftext; trailing newline weight 0",
    }
    atomic_json(args.stats_output, stats)
    print(
        "PREP_2K_COMPLETE correct_only=2000 mixed_correct=1000 mixed_wrong=1000 "
        f"stats={args.stats_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
