#!/usr/bin/env python3
"""Fail-closed validation of selections, no-system tokens, and DP schedules."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.data_utils import (
    DATASET_ROWS,
    GLOBAL_BATCH_SIZE,
    RUNTIME_PRESENTATIONS,
    SPLITS,
    UPDATES,
    VARIANTS,
    assert_record,
    atomic_json,
    normalize_token_ids,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from slime.utils.dp_schedule import build_dp_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--schedule-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def shuffled_order(size: int, seed: int, epoch: int) -> list[int]:
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def validate_schedule(rows: list[dict[str, Any]], *, seed: int, max_tokens_per_gpu: int) -> dict[str, Any]:
    lengths = [len(row["input_ids"]) for row in rows]
    order = shuffled_order(len(rows), seed, 0)
    offset = 0
    epoch = 0
    row_exposures = [0] * len(rows)
    over_cap_singletons = 0
    over_cap_exposures = 0
    maximum_mbs_tokens = 0
    microbatch_histogram: Counter[int] = Counter()
    args = SimpleNamespace(
        balance_by_flops=False,
        balance_data=False,
        max_tokens_per_gpu=max_tokens_per_gpu,
        micro_batch_size=1,
        use_dynamic_batch_size=True,
    )
    parallel = {
        "cp_size": 1,
        "dp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
        "vpp_size": 1,
    }
    for update in range(UPDATES):
        if offset + GLOBAL_BATCH_SIZE <= len(rows):
            batch_indices = order[offset : offset + GLOBAL_BATCH_SIZE]
            offset += GLOBAL_BATCH_SIZE
        else:
            batch_indices = order[offset:]
            remaining = GLOBAL_BATCH_SIZE - len(batch_indices)
            epoch += 1
            order = shuffled_order(len(rows), seed, epoch)
            batch_indices += order[:remaining]
            offset = remaining
        if len(batch_indices) != GLOBAL_BATCH_SIZE:
            raise AssertionError(f"batch-size drift at update {update}")
        for row_index in batch_indices:
            row_exposures[row_index] += 1
        batch_lengths = [lengths[index] for index in batch_indices]
        partitions, micro_batches, counts, global_sizes = build_dp_schedule(
            args,
            parallel,
            batch_lengths,
            global_batch_size=GLOBAL_BATCH_SIZE,
            rollout_indices=list(range(GLOBAL_BATCH_SIZE)),
        )
        if counts != [len(micro_batches[0])] or global_sizes != [GLOBAL_BATCH_SIZE]:
            raise ValueError(f"dynamic-schedule metadata drift at update {update}")
        placed: list[int] = []
        for local_indices in micro_batches[0]:
            positions = [partitions[0][index] for index in local_indices]
            placed.extend(positions)
            token_total = sum(batch_lengths[index] for index in positions)
            maximum_mbs_tokens = max(maximum_mbs_tokens, token_total)
            microbatch_histogram[len(positions)] += 1
            if token_total > max_tokens_per_gpu:
                if len(positions) != 1 or batch_lengths[positions[0]] <= max_tokens_per_gpu:
                    raise ValueError(f"invalid over-soft-cap microbatch at update {update}")
                over_cap_singletons += 1
        if sorted(placed) != list(range(GLOBAL_BATCH_SIZE)):
            raise ValueError(f"dynamic schedule lost or duplicated a row at update {update}")
        over_cap_exposures += sum(length > max_tokens_per_gpu for length in batch_lengths)
    expected_row_exposures = Counter({3: DATASET_ROWS - 48, 4: 48})
    if Counter(row_exposures) != expected_row_exposures:
        raise ValueError(f"282-update row exposure drift: {Counter(row_exposures)}")
    if over_cap_singletons != over_cap_exposures:
        raise ValueError("samples over the soft 16K bin cap were not scheduled alone")
    source_exposures: Counter[str] = Counter()
    for row, count in zip(rows, row_exposures):
        source_exposures[str(row["metadata"]["base_trace_id"])] += count
    if not source_exposures or min(source_exposures.values()) <= 0:
        raise ValueError("realized schedule did not cover every selected source trace")
    return {
        "dataset_rows": len(rows),
        "dataset_rows_over_soft_cap": sum(length > max_tokens_per_gpu for length in lengths),
        "effective_presentations": sum(row_exposures),
        "epochs_entered": epoch + 1,
        "final_epoch_id": epoch,
        "final_sample_offset": offset,
        "maximum_microbatch_tokens": maximum_mbs_tokens,
        "maximum_sample_tokens": max(lengths),
        "microbatch_sample_count_histogram": {str(key): value for key, value in sorted(microbatch_histogram.items())},
        "over_soft_cap_exposures": over_cap_exposures,
        "over_soft_cap_singleton_microbatches": over_cap_singletons,
        "row_exposure_histogram": {str(key): value for key, value in sorted(Counter(row_exposures).items())},
        "source_exposure_histogram": {
            str(key): value for key, value in sorted(Counter(source_exposures.values()).items())
        },
        "source_trace_count": len(source_exposures),
        "updates": UPDATES,
    }


def main() -> None:
    args = parse_args()
    contract = load_json(args.contract)
    if tuple(contract["variants"]) != VARIANTS:
        raise ValueError("contract variant order drift")
    if args.schedule_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.schedule_output}")
    training = contract["training"]
    expected_training = {
        "effective_presentations": RUNTIME_PRESENTATIONS,
        "epochs": 3,
        "final_iteration": UPDATES - 1,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "max_sequence_length": 32896,
        "max_tokens_per_gpu": 16384,
        "optimizer_updates": UPDATES,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training contract drift: {key}")
    max_length = int(training["max_sequence_length"])
    stats = load_json(args.data_root / "selection_and_tokenization_stats.json")
    sources = load_json(args.data_root / "source_artifacts.json")
    variant_summaries: dict[str, dict[str, Any]] = {}
    prefix_by_variant_problem: dict[str, dict[str, list[int]]] = {}
    schedule_reports: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        path = args.data_root / "datasets" / f"{variant}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != DATASET_ROWS:
            raise ValueError(f"{variant} row count drift: {len(rows)}")
        trace_ids = set()
        base_trace_ids = set()
        problem_ids = set()
        prefixes: dict[str, list[int]] = {}
        for row in rows:
            assert_record(row, max_length)
            metadata = row["metadata"]
            if metadata.get("variant") != variant:
                raise ValueError(f"variant metadata drift: {variant}")
            if metadata.get("train_prompt_policy") != "source_rendered_prompt_verbatim_one_user_no_system":
                raise ValueError(f"prompt policy drift: {variant}")
            trace_id = str(metadata["trace_id"])
            if trace_id in trace_ids:
                raise ValueError(f"duplicate expanded trace ID: {trace_id}")
            trace_ids.add(trace_id)
            base_trace_ids.add(str(metadata["base_trace_id"]))
            problem_id = str(metadata["problem_id"])
            problem_ids.add(problem_id)
            prefix_count = int(metadata["prefix_token_count"])
            prefix = [int(value) for value in row["input_ids"][:prefix_count]]
            if problem_id in prefixes and prefixes[problem_id] != prefix:
                raise ValueError(f"prompt prefix drift within {variant}/{problem_id}")
            prefixes[problem_id] = prefix
            expected_correct = "_correct_" in variant and "_incorrect_" not in variant
            if bool(metadata["is_correct"]) != expected_correct:
                raise ValueError(f"correctness drift in {variant}: {trace_id}")
        if sources["outputs"][variant]["sha256"] != sha256_file(path):
            raise ValueError(f"source artifact dataset hash drift: {variant}")
        if stats["expansion"][variant]["dataset_sha256"] != sha256_file(path):
            raise ValueError(f"selection stats dataset hash drift: {variant}")
        variant_summaries[variant] = {
            "base_trace_ids": base_trace_ids,
            "problem_ids": problem_ids,
        }
        prefix_by_variant_problem[variant] = prefixes
        schedule_reports[variant] = validate_schedule(
            rows,
            seed=int(contract["selection"]["seed"]),
            max_tokens_per_gpu=int(training["max_tokens_per_gpu"]),
        )
        del rows
    for split in SPLITS:
        correct_summary = variant_summaries[f"{split}_correct_6000"]
        incorrect_summary = variant_summaries[f"{split}_incorrect_6000"]
        correct_base = correct_summary["base_trace_ids"]
        incorrect_base = incorrect_summary["base_trace_ids"]
        correct_problems = correct_summary["problem_ids"]
        incorrect_problems = incorrect_summary["problem_ids"]
        if len(correct_base) != len(incorrect_base):
            raise ValueError(f"{split} condition source-trace counts differ")
        if len(correct_problems) != len(incorrect_problems):
            raise ValueError(f"{split} condition problem counts differ")
        split_stats = stats["splits"][split]
        if len(correct_base) != split_stats["correct_rows"] or len(incorrect_base) != split_stats["incorrect_rows"]:
            raise ValueError(f"{split} selected source-trace stats drift")
        if len(correct_problems) != split_stats["correct_problem_count"]:
            raise ValueError(f"{split} selected problem stats drift")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    eval_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        eval_path = args.data_root / "eval" / f"{split}_240.jsonl"
        eval_rows = read_jsonl(eval_path)
        if len(eval_rows) != 240 or len({str(row["problem_id"]) for row in eval_rows}) != 240:
            raise ValueError(f"{split} eval is not exactly 240 unique problems")
        if {str(row["split"]) for row in eval_rows} != {split}:
            raise ValueError(f"{split} eval split label drift")
        eval_by_split[split] = eval_rows
        eval_by_problem = {str(row["problem_id"]): row for row in eval_rows}
        for row in eval_rows:
            if row["prompt_sha256"] != sha256_text(row["prompt"]):
                raise ValueError(f"eval prompt hash drift: {row['problem_id']}")
            if row["rendered_prompt_sha256"] != sha256_text(row["rendered_prompt"]):
                raise ValueError(f"eval rendered-prompt hash drift: {row['problem_id']}")
        for correctness in ("correct", "incorrect"):
            variant = f"{split}_{correctness}_6000"
            for problem_id, observed_prefix in prefix_by_variant_problem[variant].items():
                eval_row = eval_by_problem[problem_id]
                prefix = normalize_token_ids(tokenizer.encode(eval_row["rendered_prompt"], add_special_tokens=False))
                if observed_prefix != prefix:
                    raise ValueError(f"training/eval prompt token drift: {split}/{problem_id}/{correctness}")
    held_in_ids = {str(row["problem_id"]) for row in eval_by_split["held_in"]}
    held_out_ids = {str(row["problem_id"]) for row in eval_by_split["held_out"]}
    if held_in_ids & held_out_ids or len(held_in_ids | held_out_ids) != 480:
        raise ValueError("held-in/held-out eval problem sets are not disjoint and exhaustive")
    if int(stats["source"]["rows"]) != 7680 or int(stats["source"]["problem_count"]) != 480:
        raise ValueError("source audit shape drift")
    if int(stats["source"]["length_stats"]["total_tokens"]["max"]) > max_length:
        raise ValueError("a source trace exceeds the hard sequence capacity")
    if not stats["invariants"]["no_length_filter_or_truncation"]:
        raise ValueError("no-length-filter invariant missing")
    schedule = {
        "artifact_schema_version": 1,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "learning_rate_schedule": {
            "maximum_lr": 5e-6,
            "minimum_lr": 1e-6,
            "style": "cosine",
            "warmup_fraction": 0.03,
            "warmup_init": 0.0,
        },
        "variants": schedule_reports,
    }
    atomic_json(args.schedule_output, schedule)
    print(
        f"AIME_SPLIT_6K_DATA_VALIDATED variants={len(VARIANTS)} " f"rows_each={DATASET_ROWS} updates={UPDATES}",
        flush=True,
    )


if __name__ == "__main__":
    main()
