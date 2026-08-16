#!/usr/bin/env python3
"""Fail-closed validation of data, prompts, exposures, and DP schedules."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.qwen2_5_7b_nous_aime_mixed_outcome_sft.data_utils import (
    DATASET_ROWS,
    GLOBAL_BATCH_SIZE,
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


def validate_schedule(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_tokens_per_gpu: int,
    updates: int,
) -> dict[str, Any]:
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
    for update in range(updates):
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
                    raise ValueError(f"invalid over-cap microbatch at update {update}")
                over_cap_singletons += 1
        if sorted(placed) != list(range(GLOBAL_BATCH_SIZE)):
            raise ValueError(f"dynamic schedule lost or duplicated a row at update {update}")
        over_cap_exposures += sum(length > max_tokens_per_gpu for length in batch_lengths)
    effective_presentations = updates * GLOBAL_BATCH_SIZE
    complete_passes, wrap_rows = divmod(effective_presentations, DATASET_ROWS)
    expected_row_exposures = Counter({complete_passes: DATASET_ROWS - wrap_rows})
    if wrap_rows:
        expected_row_exposures[complete_passes + 1] = wrap_rows
    if Counter(row_exposures) != expected_row_exposures:
        raise ValueError(f"{updates}-update row exposure drift: {Counter(row_exposures)}")
    if over_cap_singletons != over_cap_exposures:
        raise ValueError("samples over the soft 16K bin cap were not scheduled alone")
    source_exposures: Counter[str] = Counter()
    for row, count in zip(rows, row_exposures):
        source_exposures[str(row["metadata"]["base_trace_id"])] += count
    if len(source_exposures) != 181 or min(source_exposures.values()) <= 0:
        raise ValueError("realized schedule did not cover all 181 source traces")
    return {
        "dataset_rows": len(rows),
        "dataset_rows_over_16384": sum(length > max_tokens_per_gpu for length in lengths),
        "effective_presentations": sum(row_exposures),
        "epochs_entered": epoch + 1,
        "final_epoch_id": epoch,
        "final_sample_offset": offset,
        "maximum_microbatch_tokens": maximum_mbs_tokens,
        "maximum_sample_tokens": max(lengths),
        "microbatch_sample_count_histogram": {str(key): value for key, value in sorted(microbatch_histogram.items())},
        "over_16384_exposures": over_cap_exposures,
        "over_16384_singleton_microbatches": over_cap_singletons,
        "row_exposure_histogram": {str(key): value for key, value in sorted(Counter(row_exposures).items())},
        "source_exposure_histogram": {
            str(key): value for key, value in sorted(Counter(source_exposures.values()).items())
        },
        "updates": updates,
    }


def main() -> None:
    args = parse_args()
    contract = load_json(args.contract)
    if tuple(contract["variants"]) != VARIANTS:
        raise ValueError("contract variant order drift")
    if args.schedule_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.schedule_output}")
    training = contract["training"]
    updates = int(training["optimizer_updates"])
    effective_presentations = updates * GLOBAL_BATCH_SIZE
    expected_training = {
        "effective_presentations": effective_presentations,
        "final_iteration": updates - 1,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "max_sequence_length": 32768,
        "max_tokens_per_gpu": 16384,
        "optimizer_updates": updates,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training contract drift: {key}")
    max_length = int(training["max_sequence_length"])
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        path = args.data_root / "datasets" / f"{variant}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != DATASET_ROWS:
            raise ValueError(f"{variant} row count drift: {len(rows)}")
        trace_ids = set()
        for row in rows:
            assert_record(row, max_length)
            metadata = row["metadata"]
            if metadata.get("variant") != variant:
                raise ValueError(f"variant metadata drift: {variant}")
            trace_id = str(metadata["trace_id"])
            if trace_id in trace_ids:
                raise ValueError(f"duplicate expanded trace ID: {trace_id}")
            trace_ids.add(trace_id)
            expected_correct = "_correct_" in variant and "_incorrect_" not in variant
            if bool(metadata["is_correct"]) != expected_correct:
                raise ValueError(f"correctness drift in {variant}: {trace_id}")
        rows_by_variant[variant] = rows
    source_trace_counts = set()
    problem_counts = set()
    for year in ("aime24", "aime25"):
        correct = rows_by_variant[f"{year}_correct_3000"]
        incorrect = rows_by_variant[f"{year}_incorrect_3000"]
        correct_base = {str(row["metadata"]["base_trace_id"]) for row in correct}
        incorrect_base = {str(row["metadata"]["base_trace_id"]) for row in incorrect}
        if len(correct_base) != len(incorrect_base):
            raise ValueError(f"{year} condition source-trace counts differ")
        if len(correct_base) != 181:
            raise ValueError(f"{year} must contain exactly 181 source traces per condition")
        source_trace_counts.add(len(correct_base))
        correct_docs = {str(row["metadata"]["doc_id"]) for row in correct}
        incorrect_docs = {str(row["metadata"]["doc_id"]) for row in incorrect}
        if correct_docs != incorrect_docs:
            raise ValueError(f"{year} condition problem coverage differs")
        if len(correct_docs) != 10:
            raise ValueError(f"{year} must contain exactly 10 selected problems")
        problem_counts.add(len(correct_docs))
    if source_trace_counts != {181} or problem_counts != {10}:
        raise ValueError("cross-variant source-trace/problem-count controls drifted")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    for year in ("aime24", "aime25"):
        eval_path = args.data_root / "eval" / f"{year}_30.jsonl"
        eval_rows = read_jsonl(eval_path)
        if len(eval_rows) != 30 or {str(row["doc_id"]) for row in eval_rows} != {str(index) for index in range(30)}:
            raise ValueError(f"{year} evaluation rows are not exact doc_id 0..29")
        if Counter(str(row["split"]) for row in eval_rows) != Counter({"held_in": 10, "held_out": 20}):
            raise ValueError(f"{year} evaluation split must be exactly 10 held-in / 20 held-out")
        held_in = {str(row["metadata"]["doc_id"]) for row in rows_by_variant[f"{year}_correct_3000"]}
        for eval_row in eval_rows:
            expected_split = "held_in" if str(eval_row["doc_id"]) in held_in else "held_out"
            if eval_row["split"] != expected_split:
                raise ValueError(f"{year}/{eval_row['doc_id']} split drift")
            if eval_row["query_sha256"] != sha256_text(eval_row["query"]):
                raise ValueError(f"{year}/{eval_row['doc_id']} query hash drift")
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": eval_row["query"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            prefix = normalize_token_ids(prefix)
            for correctness in ("correct", "incorrect"):
                candidates = [
                    row
                    for row in rows_by_variant[f"{year}_{correctness}_3000"]
                    if str(row["metadata"]["doc_id"]) == str(eval_row["doc_id"])
                ]
                if candidates:
                    candidate = candidates[0]
                    prefix_count = int(candidate["metadata"]["prefix_token_count"])
                    if [int(value) for value in candidate["input_ids"][:prefix_count]] != prefix:
                        raise ValueError(
                            f"training/eval prompt token drift: {year}/{eval_row['doc_id']}/{correctness}"
                        )
    stats = load_json(args.data_root / "selection_and_tokenization_stats.json")
    sources = load_json(args.data_root / "source_artifacts.json")
    diagnostic = contract["selection"]["expected_post_rescore_diagnostic"]
    for year in ("aime24", "aime25"):
        year_stats = stats["years"][year]
        if year_stats["observed_differs_from_expected"]:
            raise ValueError(f"{year} realized selection differs from the pinned diagnostic")
        selection = year_stats["selection"]
        if selection["mixed_doc_ids"] != diagnostic[year]["selected_mixed_doc_ids"]:
            raise ValueError(f"{year} selected problem IDs drifted")
        if selection["full_mixed_doc_ids"] != diagnostic[year]["full_mixed_doc_ids"]:
            raise ValueError(f"{year} full mixed-problem pool drifted")
        if selection["selected_incorrect_rows"] != 181 or selection["selected_correct_rows"] != 181:
            raise ValueError(f"{year} selected source-trace count drifted")
    aime25_subset = stats["years"]["aime25"]["problem_subset_selection"]
    expected_subset = diagnostic["aime25"]
    for key, expected in (
        ("accepted_candidate_rank_zero_based", expected_subset["accepted_candidate_rank_zero_based"]),
        ("candidates_evaluated", expected_subset["candidates_evaluated"]),
        ("epsilon", 0),
        ("problem_count", 10),
        ("qualifying_candidate_count", expected_subset["qualifying_candidate_count"]),
        ("selected_delta", 0),
        ("selected_incorrect_rows", 181),
        ("target_incorrect_rows", 181),
    ):
        if aime25_subset.get(key) != expected:
            raise ValueError(f"aime25 conditioned subset audit drift: {key}")
    for variant in VARIANTS:
        path = args.data_root / "datasets" / f"{variant}.jsonl"
        if sources["outputs"][variant]["sha256"] != sha256_file(path):
            raise ValueError(f"source artifact dataset hash drift: {variant}")
        if stats["expansion"][variant]["dataset_sha256"] != sha256_file(path):
            raise ValueError(f"selection stats dataset hash drift: {variant}")
    repeat = contract.get("repeat_of")
    if repeat is not None:
        if repeat.get("contract_hash") != "2d556f01dbd853a1":
            raise ValueError("four-epoch repeat source contract drift")
        expected_datasets = repeat.get("training_dataset_sha256")
        for variant in VARIANTS:
            path = args.data_root / "datasets" / f"{variant}.jsonl"
            if not isinstance(expected_datasets, dict) or sha256_file(path) != expected_datasets.get(variant):
                raise ValueError(f"four-epoch dataset is not byte-identical: {variant}")
        expected_eval = repeat.get("evaluation_jsonl_sha256")
        for year in ("aime24", "aime25"):
            path = args.data_root / "eval" / f"{year}_30.jsonl"
            if not isinstance(expected_eval, dict) or sha256_file(path) != expected_eval.get(year):
                raise ValueError(f"four-epoch eval JSONL is not byte-identical: {year}")
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
        "variants": {
            variant: validate_schedule(
                rows,
                seed=int(contract["selection"]["seed"]),
                max_tokens_per_gpu=int(training["max_tokens_per_gpu"]),
                updates=updates,
            )
            for variant, rows in rows_by_variant.items()
        },
    }
    atomic_json(args.schedule_output, schedule)
    print(
        "NOUS_AIME_DATA_VALIDATED " f"variants={len(VARIANTS)} rows_each={DATASET_ROWS} updates={updates}",
        flush=True,
    )


if __name__ == "__main__":
    main()
