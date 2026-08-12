#!/usr/bin/env python3
"""Build nested 10%-through-90% random masks on one frozen wrong trace set."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterator

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    MASK_VARIANTS,
    assert_record,
    atomic_json,
    atomic_jsonl,
    length_summary,
    sha256_file,
    stable_unit_interval,
    trace_set_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--unmasked-data", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    return parser.parse_args()


def probability(variant: str) -> float:
    if variant not in MASK_VARIANTS:
        raise ValueError(f"unknown random-mask variant: {variant}")
    return int(variant.rsplit("_", 1)[1]) / 100.0


def variant_path(data_root: Path, variant: str) -> Path:
    return data_root / "datasets" / f"{variant}_1500.jsonl"


def masked_row(source: dict[str, Any], variant: str, seed: int) -> tuple[dict[str, Any], dict[str, int]]:
    row = copy.deepcopy(source)
    metadata = row["metadata"]
    if metadata["is_correct"] is not False:
        raise ValueError("wrong-mask source contains a correct trace")
    assistant_count = int(metadata["assistant_token_count"])
    protected = {int(index) for index in metadata["protected_assistant_positions"]}
    weights = [float(value) for value in metadata["loss_weights"]]
    if any(weights[index] != 1.0 for index in range(assistant_count)):
        raise ValueError("unmasked wrong source has a non-unit assistant weight")
    trace_id = str(metadata["trace_id"])
    threshold = probability(variant)
    masked = 0
    eligible = 0
    for index in range(assistant_count):
        if index in protected:
            continue
        eligible += 1
        if stable_unit_interval(seed, trace_id, index) < threshold:
            weights[index] = 0.0
            masked += 1
    if any(weights[index] != 1.0 for index in protected):
        raise AssertionError("protected assistant position was masked")
    if weights[assistant_count:] != [1.0, 0.0]:
        raise AssertionError("terminal supervision changed under masking")
    metadata.update(
        {
            "loss_weight_policy": (
                f"nested_random_mask_probability={threshold:.2f};" "protected_positions_bypass_draw"
            ),
            "mask_probability": threshold,
            "mask_seed": seed,
            "variant": variant,
        }
    )
    metadata["loss_weights"] = weights
    return row, {
        "eligible_tokens": eligible,
        "masked_tokens": masked,
        "protected_tokens": len(protected),
    }


def rows_for_variant(
    source: list[dict[str, Any]],
    variant: str,
    seed: int,
    aggregate: dict[str, Any],
    max_sequence_length: int,
) -> Iterator[dict[str, Any]]:
    trace_ids: list[str] = []
    for source_row in source:
        row, counts = masked_row(source_row, variant, seed)
        assert_record(row, max_sequence_length)
        trace_ids.append(str(row["metadata"]["trace_id"]))
        aggregate["rows"] += 1
        for key, value in counts.items():
            aggregate[key] += value
        yield row
    aggregate["ordered_trace_ids_sha256"] = trace_set_sha256(trace_ids)


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    seed = int(contract["masking"]["seed"])
    expected_probabilities = [probability(variant) for variant in MASK_VARIANTS]
    if expected_probabilities != contract["masking"]["probabilities"]:
        raise ValueError("contract/random-mask probability drift")
    outputs = {variant: variant_path(args.data_root, variant) for variant in MASK_VARIANTS}
    existing = [str(path) for path in [*outputs.values(), args.stats_output] if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite random-mask artifacts: {existing}")
    with args.unmasked_data.open(encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    if len(source) != 1500:
        raise ValueError(f"expected 1,500 frozen wrong rows, got {len(source)}")
    trace_ids = [str(row["metadata"]["trace_id"]) for row in source]
    if len(set(trace_ids)) != 1500:
        raise ValueError("frozen wrong trace IDs are not unique")

    max_sequence_length = int(contract["training"]["max_sequence_length"])
    source_lengths = length_summary(source)
    stats: dict[str, Any] = {
        "artifact_schema_version": 1,
        "mask_seed": seed,
        "nested_by_shared_stable_draw": True,
        "ordered_wrong_trace_ids_sha256": trace_set_sha256(trace_ids),
        "protected_policy": (
            "all tokenizer added/control and literal think-tag positions remain weight 1; "
            "im_end remains 1; final newline remains 0"
        ),
        "source_length_stats": source_lengths,
        "source_unmasked_sha256": sha256_file(args.unmasked_data),
        "variants": {},
    }
    for variant, output in outputs.items():
        aggregate = {
            "eligible_tokens": 0,
            "masked_tokens": 0,
            "protected_tokens": 0,
            "rows": 0,
        }
        atomic_jsonl(
            output,
            rows_for_variant(
                source,
                variant,
                seed,
                aggregate,
                max_sequence_length,
            ),
        )
        if aggregate["rows"] != 1500:
            raise ValueError(f"random-mask row-count drift for {variant}")
        aggregate.update(
            {
                "actual_mask_fraction": (aggregate["masked_tokens"] / aggregate["eligible_tokens"]),
                "dataset_sha256": sha256_file(output),
                "mask_probability": probability(variant),
            }
        )
        stats["variants"][variant] = aggregate
        print(
            f"RANDOM_MASK_BUILD_COMPLETE variant={variant} " f"actual={aggregate['actual_mask_fraction']:.8f}",
            flush=True,
        )
    atomic_json(args.stats_output, stats)


if __name__ == "__main__":
    main()
