#!/usr/bin/env python3
"""Retokenize the frozen Axolotl 2K selections for Qwen2.5-7B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    CHAT_BOUNDARY_LITERALS,
    atomic_json,
    atomic_jsonl,
    build_training_record,
    sha256_file,
    source_location,
    token_sha256,
    tokenizer_control_inventory,
    trace_set_sha256,
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
    parser.add_argument("--inventory-output", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument(
        "--chat-template-kwargs",
        default="{}",
        help="JSON object forwarded to tokenizer.apply_chat_template",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_trace_structure(trace: str) -> tuple[bool, str]:
    if any(token in trace for token in CHAT_BOUNDARY_LITERALS):
        return False, "embedded_chat_control"
    if trace.count("<think>") != 1:
        return False, "opening_think_tag_count"
    start = trace.index("<think>") + len("<think>")
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
    if not closes:
        return ((True, "missing_close_preserved") if trace[start:].strip() else (False, "empty_after_open"))
    if not trace[closes[-1] + len("</think>") :].strip():
        return False, "empty_post_think"
    return ((True, "ok") if len(closes) == 1 else (True, "duplicate_close_preserved"))


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
    assistant = str(messages[1]["content"])
    valid, reason = canonical_trace_structure(assistant)
    if not valid:
        raise ValueError(f"invalid assistant structure for {trace_id}: {reason}")
    assistant_ids = [
        int(value) for value in tokenizer.encode(assistant, add_special_tokens=False)
    ]
    return {
        "answer": str(metadata["correct_answer"]),
        "assistant_token_count": len(assistant_ids),
        "assistant_token_sha256": token_sha256(assistant_ids),
        "assistant_trace": assistant,
        "closing_think_tag_count": assistant.count("</think>"),
        "correctness_source": "axolotl_metadata.is_correct",
        "generation_index": generation,
        "is_correct": bool(metadata["is_correct"]),
        "problem": user_text.split(marker, 1)[1],
        "problem_id": str(metadata["problem_id"]),
        "source_row_index": source_row,
        "trace_id": trace_id,
    }


def main() -> None:
    args = parse_args()
    chat_template_kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("--chat-template-kwargs must decode to a JSON object")
    destinations = (
        args.selected_output,
        args.correct_output,
        args.mixed_output,
        args.stats_output,
        args.inventory_output,
    )
    for destination in destinations:
        if Path(destination).exists():
            raise FileExistsError(f"refusing to overwrite {destination}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    correct = [normalize_source(row, tokenizer) for row in load_jsonl(args.source_correct_only)]
    mixed = [normalize_source(row, tokenizer) for row in load_jsonl(args.source_mixed_unmasked)]
    if len(correct) != 2000 or len(mixed) != 2000:
        raise ValueError(f"canonical row count drift: correct={len(correct)} mixed={len(mixed)}")
    if any(not row["is_correct"] for row in correct):
        raise ValueError("correct-only source contains a wrong trace")
    mixed_correct = [row for row in mixed if row["is_correct"]]
    mixed_wrong = [row for row in mixed if not row["is_correct"]]
    if (len(mixed_correct), len(mixed_wrong)) != (1000, 1000):
        raise ValueError("mixed source is not 1000 correct plus 1000 wrong")
    if {row["problem_id"] for row in mixed_correct} & {row["problem_id"] for row in mixed_wrong}:
        raise ValueError("mixed source is paired; expected disjoint unpaired prompts")
    correct_by_trace = {row["trace_id"]: row for row in correct}
    for row in mixed_correct:
        if correct_by_trace.get(row["trace_id"]) != row:
            raise ValueError(f"mixed correct trace is not the frozen correct subset: {row['trace_id']}")
    for wrong_index, row in enumerate(mixed_wrong):
        row["wrong_index"] = wrong_index
    selected_union = correct + mixed_wrong
    if len({row["trace_id"] for row in selected_union}) != 3000:
        raise ValueError("selected union contains duplicate traces")

    correct_records = (
        build_training_record(
            tokenizer,
            row,
            None,
            args.max_sequence_length,
            chat_template_kwargs,
        )
        for row in correct
    )
    mixed_records = (
        build_training_record(
            tokenizer,
            row,
            None,
            args.max_sequence_length,
            chat_template_kwargs,
        )
        for row in mixed
    )
    atomic_jsonl(args.selected_output, selected_union)
    atomic_jsonl(args.correct_output, correct_records)
    atomic_jsonl(args.mixed_output, mixed_records)
    atomic_json(args.inventory_output, tokenizer_control_inventory(tokenizer))
    stats = {
        "counts": {
            "correct_only": len(correct),
            "mixed_correct": len(mixed_correct),
            "mixed_wrong": len(mixed_wrong),
            "selected_union": len(selected_union),
        },
        "trace_sets": {
            "correct_only_ordered_sha256": trace_set_sha256([row["trace_id"] for row in correct]),
            "mixed_ordered_sha256": trace_set_sha256([row["trace_id"] for row in mixed]),
        },
        "source_artifacts": {
            "correct_only_sha256": sha256_file(args.source_correct_only),
            "correct_wrong_unmasked_sha256": sha256_file(args.source_mixed_unmasked),
        },
        "source_structure": {
            "correct_only_missing_close": sum(row["closing_think_tag_count"] == 0 for row in correct),
            "correct_only_duplicate_close": sum(row["closing_think_tag_count"] > 1 for row in correct),
            "mixed_missing_close": sum(row["closing_think_tag_count"] == 0 for row in mixed),
            "mixed_duplicate_close": sum(row["closing_think_tag_count"] > 1 for row in mixed),
        },
        "output_artifacts": {
            "selected_union_sha256": sha256_file(args.selected_output),
            "correct_only_sha256": sha256_file(args.correct_output),
            "mixed_unmasked_sha256": sha256_file(args.mixed_output),
        },
        "prompt_policy": (
            "Qwen2.5 native chat template without enable_thinking"
            if not chat_template_kwargs
            else {
                "apply_chat_template_kwargs": chat_template_kwargs,
                "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            }
        ),
        "supervision_policy": "all added/control tokens and literal think markup bypass masking; im_end exact weight 1",
    }
    atomic_json(args.stats_output, stats)
    print("PREP_QWEN25_2K_COMPLETE correct=2000 mixed_correct=1000 mixed_wrong=1000", flush=True)


if __name__ == "__main__":
    main()
