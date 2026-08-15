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
    RUNTIME_PRESENTATIONS,
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
                    raise ValueError(f"invalid over-cap microbatch at update {update}")
                over_cap_singletons += 1
        if sorted(placed) != list(range(GLOBAL_BATCH_SIZE)):
            raise ValueError(f"dynamic schedule lost or duplicated a row at update {update}")
        over_cap_exposures += sum(length > max_tokens_per_gpu for length in batch_lengths)
    if Counter(row_exposures) != Counter({1: DATASET_ROWS - 8, 2: 8}):
        raise ValueError(f"47-update row exposure drift: {Counter(row_exposures)}")
    if over_cap_singletons != over_cap_exposures:
        raise ValueError("samples over the soft 16K bin cap were not scheduled alone")
    source_exposures: Counter[str] = Counter()
    for row, count in zip(rows, row_exposures):
        source_exposures[str(row["metadata"]["base_trace_id"])] += count
    if max(source_exposures.values()) - min(source_exposures.values()) != 1:
        raise ValueError("realized source-trace exposure differs by more than one")
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
        "final_iteration": 46,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "max_sequence_length": 32768,
        "max_tokens_per_gpu": 16384,
        "optimizer_updates": UPDATES,
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
    for year in ("aime24", "aime25"):
        correct = rows_by_variant[f"{year}_correct_3000"]
        incorrect = rows_by_variant[f"{year}_incorrect_3000"]
        correct_base = {str(row["metadata"]["base_trace_id"]) for row in correct}
        incorrect_base = {str(row["metadata"]["base_trace_id"]) for row in incorrect}
        if len(correct_base) != len(incorrect_base):
            raise ValueError(f"{year} condition source-trace counts differ")
        correct_docs = {str(row["metadata"]["doc_id"]) for row in correct}
        incorrect_docs = {str(row["metadata"]["doc_id"]) for row in incorrect}
        if correct_docs != incorrect_docs:
            raise ValueError(f"{year} condition problem coverage differs")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    for year in ("aime24", "aime25"):
        eval_path = args.data_root / "eval" / f"{year}_30.jsonl"
        eval_rows = read_jsonl(eval_path)
        if len(eval_rows) != 30 or {str(row["doc_id"]) for row in eval_rows} != {str(index) for index in range(30)}:
            raise ValueError(f"{year} evaluation rows are not exact doc_id 0..29")
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
    for variant in VARIANTS:
        path = args.data_root / "datasets" / f"{variant}.jsonl"
        if sources["outputs"][variant]["sha256"] != sha256_file(path):
            raise ValueError(f"source artifact dataset hash drift: {variant}")
        if stats["expansion"][variant]["dataset_sha256"] != sha256_file(path):
            raise ValueError(f"selection stats dataset hash drift: {variant}")
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
            )
            for variant, rows in rows_by_variant.items()
        },
    }
    atomic_json(args.schedule_output, schedule)
    print(
        "NOUS_AIME_DATA_VALIDATED " f"variants={len(VARIANTS)} rows_each={DATASET_ROWS} updates={UPDATES}",
        flush=True,
    )


if __name__ == "__main__":
    main()
