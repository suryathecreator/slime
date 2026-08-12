#!/usr/bin/env python3
"""Fail-closed validation of all prepared AIME/OpenR1 training datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    CHILD_VARIANTS,
    MASK_VARIANTS,
    assert_record,
    atomic_json,
    sha256_file,
    stable_unit_interval,
    trace_set_sha256,
)
from slime.utils.dp_schedule import build_dp_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--schedule-output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def data_path(root: Path, variant: str) -> Path:
    names = {
        "aime_correct_3000": "aime_correct_3000.jsonl",
        "continue_wrong_unmasked": "continue_wrong_unmasked_1500.jsonl",
        "continue_correct_only_1500": "continue_correct_only_1500.jsonl",
    }
    if variant in MASK_VARIANTS:
        return root / "datasets" / f"{variant}_1500.jsonl"
    return root / "datasets" / names[variant]


def validate_base_set(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_correct: bool,
    max_sequence_length: int,
) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")
    trace_ids: list[str] = []
    for row in rows:
        assert_record(row, max_sequence_length)
        metadata = row["metadata"]
        if metadata["is_correct"] is not expected_correct:
            raise ValueError("dataset correctness-label drift")
        trace_ids.append(str(metadata["trace_id"]))
    if len(set(trace_ids)) != expected_rows:
        raise ValueError("dataset trace IDs are not unique")


def validate_stage1(rows: list[dict[str, Any]], max_sequence_length: int) -> None:
    validate_base_set(
        rows,
        expected_rows=3000,
        expected_correct=True,
        max_sequence_length=max_sequence_length,
    )
    copies: dict[str, list[int]] = {}
    for row in rows:
        metadata = row["metadata"]
        copies.setdefault(str(metadata["problem_id"]), []).append(int(metadata["copy_index"]))
        if metadata["variant"] != "aime_correct_3000":
            raise ValueError("stage1 variant metadata drift")
        if any(value != 1.0 for value in metadata["loss_weights"][:-1]):
            raise ValueError("stage1 has a masked supervised token")
    multiplicities = sorted(len(values) for values in copies.values())
    if multiplicities != [7] * 200 + [8] * 200:
        raise ValueError("stage1 7/8-copy exposure drift")
    if any(sorted(values) not in (list(range(7)), list(range(8))) for values in copies.values()):
        raise ValueError("stage1 copy-index drift")


def validate_random_masks(
    unmasked: list[dict[str, Any]],
    variants: dict[str, list[dict[str, Any]]],
    seed: int,
    max_sequence_length: int,
) -> None:
    previous_masked: list[set[int]] | None = None
    base_trace_ids = [str(row["metadata"]["trace_id"]) for row in unmasked]
    for variant in MASK_VARIANTS:
        rows = variants[variant]
        if len(rows) != len(unmasked):
            raise ValueError(f"row count drift for {variant}")
        threshold = int(variant.rsplit("_", 1)[1]) / 100.0
        current_masked: list[set[int]] = []
        for source, row, expected_trace_id in zip(unmasked, rows, base_trace_ids, strict=True):
            assert_record(row, max_sequence_length)
            source_metadata = source["metadata"]
            metadata = row["metadata"]
            if metadata["trace_id"] != expected_trace_id:
                raise ValueError(f"trace-order drift for {variant}")
            if row["input_ids"] != source["input_ids"]:
                raise ValueError(f"token IDs changed for {variant}:{expected_trace_id}")
            assistant_count = int(metadata["assistant_token_count"])
            protected = {int(index) for index in metadata["protected_assistant_positions"]}
            source_weights = source_metadata["loss_weights"]
            weights = metadata["loss_weights"]
            masked: set[int] = set()
            for index in range(assistant_count):
                expected = 1.0
                if index not in protected:
                    expected = float(stable_unit_interval(seed, expected_trace_id, index) >= threshold)
                if weights[index] != expected:
                    raise ValueError(f"random draw mismatch {variant}:{expected_trace_id}:{index}")
                if expected == 0.0:
                    masked.add(index)
            if weights[assistant_count:] != [1.0, 0.0]:
                raise ValueError("terminal weights changed under masking")
            if source_weights[assistant_count:] != [1.0, 0.0]:
                raise ValueError("unmasked source terminal weights drifted")
            current_masked.append(masked)
        if previous_masked is not None:
            for previous, current in zip(previous_masked, current_masked, strict=True):
                if not previous <= current:
                    raise ValueError("random masks are not nested by threshold")
        previous_masked = current_masked


def shuffled_order(size: int, seed: int, epoch: int) -> list[int]:
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def realized_dynamic_schedule(
    rows: list[dict[str, Any]],
    *,
    updates: int,
    seed: int,
    global_batch_size: int,
    max_tokens_per_gpu: int,
) -> dict[str, Any]:
    """Run the actual Slime first-fit DP scheduler over every intended update."""
    lengths = [len(row["input_ids"]) for row in rows]
    size = len(lengths)
    epoch = 0
    offset = 0
    order = shuffled_order(size, seed, epoch)
    exposure_counts = [0] * size
    per_rank_mbs_histogram: Counter[int] = Counter()
    mbs_sample_count_histogram: Counter[int] = Counter()
    over_cap_singletons = 0
    over_cap_exposures = 0
    maximum_mbs_tokens = 0
    args = SimpleNamespace(
        balance_by_flops=False,
        balance_data=False,
        max_tokens_per_gpu=max_tokens_per_gpu,
        micro_batch_size=1,
        use_dynamic_batch_size=True,
    )
    parallel = {
        "cp_size": 1,
        "dp_size": 2,
        "microbatch_group_size_per_vp_stage": 1,
        "vpp_size": 1,
    }
    for update in range(updates):
        if offset + global_batch_size <= size:
            batch_rows = order[offset : offset + global_batch_size]
            offset += global_batch_size
        else:
            batch_rows = order[offset:]
            remaining = global_batch_size - len(batch_rows)
            epoch += 1
            order = shuffled_order(size, seed, epoch)
            batch_rows += order[:remaining]
            offset = remaining
        if len(batch_rows) != global_batch_size:
            raise AssertionError(f"rollout batch-size drift at update {update}")
        for row_index in batch_rows:
            exposure_counts[row_index] += 1
        batch_lengths = [lengths[index] for index in batch_rows]
        partitions, micro_batches, counts, global_sizes = build_dp_schedule(
            args,
            parallel,
            batch_lengths,
            global_batch_size=global_batch_size,
            rollout_indices=list(range(global_batch_size)),
        )
        if counts != [len(micro_batches[0])] or global_sizes != [global_batch_size]:
            raise ValueError(f"realized schedule metadata drift at update {update}")
        if len(micro_batches[0]) != len(micro_batches[1]):
            raise ValueError(f"unequal DP microbatch counts at update {update}")
        per_rank_mbs_histogram[len(micro_batches[0])] += 1
        placed: list[int] = []
        for rank in range(2):
            for local_indices in micro_batches[rank]:
                batch_positions = [partitions[rank][index] for index in local_indices]
                placed.extend(batch_positions)
                token_total = sum(batch_lengths[index] for index in batch_positions)
                maximum_mbs_tokens = max(maximum_mbs_tokens, token_total)
                mbs_sample_count_histogram[len(batch_positions)] += 1
                if token_total > max_tokens_per_gpu:
                    if len(batch_positions) != 1 or batch_lengths[batch_positions[0]] <= max_tokens_per_gpu:
                        raise ValueError(f"non-singleton or invalid over-cap mbs at update {update}")
                    over_cap_singletons += 1
        if sorted(placed) != list(range(global_batch_size)):
            raise ValueError(f"DP schedule lost or duplicated a sample at update {update}")
        over_cap_exposures += sum(length > max_tokens_per_gpu for length in batch_lengths)
    if over_cap_singletons != over_cap_exposures:
        raise ValueError("over-16K samples were not always scheduled as lone mbs")
    exposure_histogram = Counter(exposure_counts)
    expected_five = updates * global_batch_size - 4 * size
    expected_exposure_histogram = {4: size - expected_five, 5: expected_five}
    if dict(sorted(exposure_histogram.items())) != expected_exposure_histogram:
        raise ValueError(
            "four-epoch exposure/wrap drift: " f"expected={expected_exposure_histogram} actual={exposure_histogram}"
        )
    return {
        "dataset_rows": size,
        "dataset_rows_over_16384": sum(length > max_tokens_per_gpu for length in lengths),
        "epochs_entered": epoch + 1,
        "exposure_histogram": {str(key): value for key, value in sorted(exposure_histogram.items())},
        "exposures": sum(exposure_counts),
        "final_epoch_id": epoch,
        "final_sample_offset": offset,
        "global_batch_size": global_batch_size,
        "maximum_mbs_tokens": maximum_mbs_tokens,
        "maximum_sample_tokens": max(lengths),
        "mbs_sample_count_histogram": {str(key): value for key, value in sorted(mbs_sample_count_histogram.items())},
        "over_16384_exposures": over_cap_exposures,
        "over_16384_singleton_mbs": over_cap_singletons,
        "per_rank_mbs_count_per_update_histogram": {
            str(key): value for key, value in sorted(per_rank_mbs_histogram.items())
        },
        "rollout_shuffle_policy": "random.Random(seed + epoch_id).shuffle(origin_indices)",
        "updates": updates,
    }


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if args.schedule_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.schedule_output}")
    max_sequence_length = int(contract["training"]["max_sequence_length"])
    stage1_path = data_path(args.data_root, "aime_correct_3000")
    wrong_path = data_path(args.data_root, "continue_wrong_unmasked")
    correct_path = data_path(args.data_root, "continue_correct_only_1500")
    stage1 = load(stage1_path)
    wrong = load(wrong_path)
    correct = load(correct_path)
    validate_stage1(stage1, max_sequence_length)
    validate_base_set(
        wrong,
        expected_rows=1500,
        expected_correct=False,
        max_sequence_length=max_sequence_length,
    )
    validate_base_set(
        correct,
        expected_rows=1500,
        expected_correct=True,
        max_sequence_length=max_sequence_length,
    )
    wrong_problem_ids = {str(row["metadata"]["problem_id"]) for row in wrong}
    correct_problem_ids = {str(row["metadata"]["problem_id"]) for row in correct}
    if wrong_problem_ids & correct_problem_ids:
        raise ValueError("OpenR1 correct/wrong problem IDs overlap")
    variants = {variant: load(data_path(args.data_root, variant)) for variant in MASK_VARIANTS}
    validate_random_masks(
        wrong,
        variants,
        int(contract["masking"]["seed"]),
        max_sequence_length,
    )
    stats_path = args.data_root / "selection_and_tokenization_stats.json"
    mask_stats_path = args.data_root / "variant_stats.json"
    source_artifacts_path = args.data_root / "source_artifacts.json"
    for required in (stats_path, mask_stats_path, source_artifacts_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing preparation evidence: {required}")
    mask_stats = json.loads(mask_stats_path.read_text(encoding="utf-8"))
    base_trace_hash = trace_set_sha256([str(row["metadata"]["trace_id"]) for row in wrong])
    if mask_stats["ordered_wrong_trace_ids_sha256"] != base_trace_hash:
        raise ValueError("variant-stats trace-order drift")
    for variant in MASK_VARIANTS:
        path = data_path(args.data_root, variant)
        if mask_stats["variants"][variant]["dataset_sha256"] != sha256_file(path):
            raise ValueError(f"variant dataset hash drift: {variant}")

    training = contract["training"]
    if tuple(contract["variants"][1:]) != CHILD_VARIANTS:
        raise ValueError("contract child order drift")
    schedule = {
        "artifact_schema_version": 1,
        "global_batch_size": 64,
        "learning_rate_schedule": {
            "maximum_lr": 5e-6,
            "minimum_lr": 1e-6,
            "style": "cosine",
            "warmup_fraction": 0.03,
            "warmup_init": 0.0,
        },
        "stages": {
            "aime_correct_3000": {
                "dataset_rows": 3000,
                "effective_examples_exposed": 188 * 64,
                "final_iteration": 187,
                "lr_decay_iters": 188,
                "optimizer_updates": 188,
                "realized_dynamic_schedule": realized_dynamic_schedule(
                    stage1,
                    updates=188,
                    seed=int(contract["selection"]["aime"]["seed"]),
                    global_batch_size=64,
                    max_tokens_per_gpu=16384,
                ),
            },
            "children": {
                "dataset_rows_each": 1500,
                "effective_examples_exposed_each": 94 * 64,
                "final_iteration": 93,
                "fresh_optimizer_from_stage1_hf": True,
                "lr_decay_iters": 94,
                "optimizer_updates_each": 94,
                "variants": list(CHILD_VARIANTS),
                "realized_dynamic_schedule_correct_only": realized_dynamic_schedule(
                    correct,
                    updates=94,
                    seed=int(contract["selection"]["openr1"]["seed"]),
                    global_batch_size=64,
                    max_tokens_per_gpu=16384,
                ),
                "realized_dynamic_schedule_wrong_shared_by_masks": realized_dynamic_schedule(
                    wrong,
                    updates=94,
                    seed=int(contract["selection"]["openr1"]["seed"]),
                    global_batch_size=64,
                    max_tokens_per_gpu=16384,
                ),
            },
        },
    }
    expected = {
        "child_final_iteration": 93,
        "child_optimizer_updates": 94,
        "global_batch_size": 64,
        "stage1_final_iteration": 187,
        "stage1_optimizer_updates": 188,
    }
    for key, value in expected.items():
        if training[key] != value:
            raise ValueError(f"training schedule contract drift: {key}")
    atomic_json(args.schedule_output, schedule)
    print(
        "AIME_GENERALIZATION_DATA_VALIDATED "
        f"stage1={len(stage1)} children={len(CHILD_VARIANTS)} wrong={len(wrong)} "
        f"correct={len(correct)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
