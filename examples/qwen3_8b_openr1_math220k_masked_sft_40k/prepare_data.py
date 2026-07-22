#!/usr/bin/env python3
"""Select canonical OpenR1 traces and build exact-token SFT datasets."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    atomic_json,
    atomic_jsonl,
    build_training_record,
    chat_prefix_ids,
    chat_prefix_suffix_ids,
    correctness_at,
    problem_id,
    sha256_file,
    token_sha256,
    trace_structure_ok,
    usable_text,
    user_content,
    value_at,
)


def parse_args() -> argparse.Namespace:
    """Parse the pinned data-preparation contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument("--correct-only-rows", type=int, default=40000)
    parser.add_argument("--mixed-correct-rows", type=int, default=20000)
    parser.add_argument("--mixed-wrong-rows", type=int, default=20000)
    parser.add_argument("--selected-traces", required=True)
    parser.add_argument("--correct-only-output", required=True)
    parser.add_argument("--unmasked-output", required=True)
    parser.add_argument("--stats-output", required=True)
    return parser.parse_args()


def deterministic_order(indices: list[int], seed: int, row_index: int, label: str) -> list[int]:
    """Shuffle generation choices reproducibly for one source problem."""
    local = random.Random(f"{seed}:{row_index}:{label}")
    result = list(indices)
    local.shuffle(result)
    return result


def eligible_ref(
    row: dict[str, Any],
    row_index: int,
    want_correct: bool,
    tokenizer: Any,
    args: argparse.Namespace,
    counts: Counter,
) -> dict[str, Any] | None:
    """Choose one seeded fitting trace for a problem and correctness class."""
    problem = usable_text(row.get("problem"))
    answer = usable_text(row.get("answer"))
    generations = row.get("generations")
    if problem is None or answer is None or not isinstance(generations, (list, tuple)):
        counts["row_missing_required_fields"] += 1
        return None
    candidate_indices: list[int] = []
    source_by_index: dict[int, str] = {}
    for index in range(len(generations)):
        correct, source = correctness_at(row, index)
        if correct is not want_correct:
            continue
        if value_at(row.get("is_reasoning_complete"), index) is not True:
            counts["trace_incomplete_source_flag"] += 1
            continue
        trace = usable_text(generations[index])
        if trace is None:
            counts["trace_empty"] += 1
            continue
        structure_ok, reason = trace_structure_ok(trace)
        if not structure_ok:
            counts[f"trace_rejected_{reason}"] += 1
            continue
        candidate_indices.append(index)
        source_by_index[index] = str(source)

    label = "correct" if want_correct else "wrong"
    for generation_index in deterministic_order(candidate_indices, args.seed, row_index, label):
        trace = str(generations[generation_index]).strip()
        assistant_ids = [int(token) for token in tokenizer.encode(trace, add_special_tokens=False)]
        if not assistant_ids:
            counts["trace_empty_tokens"] += 1
            continue
        prefix, suffix = chat_prefix_suffix_ids(tokenizer, user_content(problem))
        if len(prefix) + len(assistant_ids) + len(suffix) > args.max_sequence_length:
            counts["trace_too_long_training"] += 1
            continue
        if not want_correct:
            conditioned = chat_prefix_ids(tokenizer, user_content(problem, answer))
            if len(conditioned) + len(assistant_ids) > args.max_sequence_length:
                counts["trace_too_long_conditioned"] += 1
                continue
        pid = problem_id(row, row_index)
        return {
            "assistant_token_count": len(assistant_ids),
            "assistant_token_sha256": token_sha256(assistant_ids),
            "correctness_source": source_by_index[generation_index],
            "finish_reason": value_at(row.get("finish_reasons"), generation_index),
            "generation_index": generation_index,
            "is_correct": want_correct,
            "problem_id": pid,
            "source_row_index": row_index,
            "trace_id": f"{pid}__row_{row_index:07d}__gen_{generation_index:03d}",
        }
    counts[f"problem_no_eligible_{label}"] += 1
    return None


def reservoir_select(
    dataset: Any,
    target: int,
    want_correct: bool,
    excluded_problem_ids: set[str],
    tokenizer: Any,
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], Counter, int]:
    """Uniformly reservoir-sample one eligible trace per unique problem."""
    rng = random.Random(args.seed + seed_offset)
    reservoir: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen = 0
    seen_problem_ids: set[str] = set()
    for row_index, row in enumerate(dataset):
        counts["source_rows_seen"] += 1
        pid = problem_id(row, row_index)
        if pid in excluded_problem_ids or pid in seen_problem_ids:
            counts["problem_excluded_or_duplicate"] += 1
            continue
        seen_problem_ids.add(pid)
        ref = eligible_ref(row, row_index, want_correct, tokenizer, args, counts)
        if ref is None:
            continue
        seen += 1
        if len(reservoir) < target:
            reservoir.append(ref)
        else:
            slot = rng.randrange(seen)
            if slot < target:
                reservoir[slot] = ref
        if counts["source_rows_seen"] % 5000 == 0:
            print(
                f"PREP_PROGRESS class={'correct' if want_correct else 'wrong'} "
                f"rows={counts['source_rows_seen']} eligible={seen} reservoir={len(reservoir)}",
                flush=True,
            )
    if len(reservoir) != target:
        raise RuntimeError(f"wanted {target} {'correct' if want_correct else 'wrong'} traces, found {len(reservoir)}")
    return reservoir, counts, seen


def materialize(dataset: Any, ref: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Reconstruct and verify a selected trace from its source row reference."""
    row = dataset[int(ref["source_row_index"])]
    trace = str(row["generations"][int(ref["generation_index"])]).strip()
    assistant_ids = [int(token) for token in tokenizer.encode(trace, add_special_tokens=False)]
    if len(assistant_ids) != int(ref["assistant_token_count"]):
        raise ValueError(f"token count drift for {ref['trace_id']}")
    if token_sha256(assistant_ids) != ref["assistant_token_sha256"]:
        raise ValueError(f"token hash drift for {ref['trace_id']}")
    result = dict(ref)
    result.update(
        {
            "answer": str(row["answer"]),
            "assistant_trace": trace,
            "problem": str(row["problem"]),
            "source": row.get("source"),
            "source_is_reasoning_complete": True,
        }
    )
    return result


def main() -> None:
    """Run selection, tokenization, and unweighted dataset construction."""
    args = parse_args()
    outputs = [args.selected_traces, args.correct_only_output, args.unmasked_output, args.stats_output]
    existing = [path for path in outputs if Path(path).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite preparation artifacts: {existing}")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    correct_refs, correct_counts, correct_seen = reservoir_select(
        dataset, args.correct_only_rows, True, set(), tokenizer, args, seed_offset=1000
    )
    correct_problem_ids = {str(ref["problem_id"]) for ref in correct_refs}
    if len(correct_problem_ids) != args.correct_only_rows:
        raise RuntimeError("correct-only problem IDs are not unique")
    wrong_refs, wrong_counts, wrong_seen = reservoir_select(
        dataset,
        args.mixed_wrong_rows,
        False,
        correct_problem_ids,
        tokenizer,
        args,
        seed_offset=2000,
    )
    wrong_problem_ids = {str(ref["problem_id"]) for ref in wrong_refs}
    if correct_problem_ids & wrong_problem_ids:
        raise RuntimeError("correct and wrong problem IDs overlap")

    mixed_rng = random.Random(args.seed + 3000)
    mixed_correct_refs = mixed_rng.sample(correct_refs, args.mixed_correct_rows)
    correct_order = list(correct_refs)
    random.Random(args.seed + 4000).shuffle(correct_order)
    mixed_order = list(mixed_correct_refs) + list(wrong_refs)
    random.Random(args.seed + 5000).shuffle(mixed_order)

    selected_by_id: dict[str, dict[str, Any]] = {}
    mixed_correct_trace_ids = {str(ref["trace_id"]) for ref in mixed_correct_refs}
    wrong_index_by_trace = {str(ref["trace_id"]): index for index, ref in enumerate(wrong_refs)}
    for ref in correct_refs + wrong_refs:
        row = materialize(dataset, ref, tokenizer)
        if not bool(ref["is_correct"]):
            row["wrong_index"] = wrong_index_by_trace[str(ref["trace_id"])]
        row["in_correct_only"] = bool(ref["is_correct"])
        row["in_mixed"] = str(ref["trace_id"]) in mixed_correct_trace_ids or not bool(ref["is_correct"])
        selected_by_id[str(ref["trace_id"])] = row
    if len(selected_by_id) != args.correct_only_rows + args.mixed_wrong_rows:
        raise RuntimeError("selected trace IDs are not unique")

    selected_rows = [selected_by_id[str(ref["trace_id"])] for ref in correct_refs + wrong_refs]
    atomic_jsonl(args.selected_traces, selected_rows)
    atomic_jsonl(
        args.correct_only_output,
        (
            build_training_record(tokenizer, selected_by_id[str(ref["trace_id"])], None, args.max_sequence_length)
            for ref in correct_order
        ),
    )
    atomic_jsonl(
        args.unmasked_output,
        (
            build_training_record(tokenizer, selected_by_id[str(ref["trace_id"])], None, args.max_sequence_length)
            for ref in mixed_order
        ),
    )

    stats = {
        "contract": {
            "correct_only_rows": args.correct_only_rows,
            "dataset": args.dataset,
            "dataset_config": args.config,
            "dataset_revision": args.revision,
            "max_sequence_length": args.max_sequence_length,
            "mixed_correct_rows": args.mixed_correct_rows,
            "mixed_wrong_rows": args.mixed_wrong_rows,
            "model": args.model,
            "seed": args.seed,
            "split": args.split,
        },
        "correct_selection": {"eligible_problems_seen": correct_seen, "skip_counts": dict(correct_counts)},
        "wrong_selection": {"eligible_problems_seen": wrong_seen, "skip_counts": dict(wrong_counts)},
        "invariants": {
            "correct_only_unique_problem_ids": len(correct_problem_ids),
            "mixed_correct_is_subset_of_correct_only": True,
            "mixed_correct_wrong_problem_overlap": len(correct_problem_ids & wrong_problem_ids),
            "source_completion_flag_required": True,
            "think_and_im_end_supervised": True,
            "synthetic_endoftext_appended": False,
        },
        "artifacts": {},
    }
    for name, path in (
        ("selected_traces", args.selected_traces),
        ("correct_only", args.correct_only_output),
        ("unmasked", args.unmasked_output),
    ):
        stats["artifacts"][name] = {"path": path, "sha256": sha256_file(path)}
    atomic_json(args.stats_output, stats)
    print(f"PREP_COMPLETE stats={args.stats_output}", flush=True)


if __name__ == "__main__":
    main()
