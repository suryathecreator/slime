#!/usr/bin/env python3
"""Recompute and verify one low-rate random-mask dataset exactly."""

from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path

from examples.qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k.build_random_masks import (
    load_variant_specs,
    wrong_assistant_weights,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    trace_set_sha256,
)

GENERATED_METADATA = {
    "loss_weight_policy",
    "mask_probability",
    "mask_seed",
    "variant",
}


def normalized_source_metadata(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items() if key not in GENERATED_METADATA}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-unmasked", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--expected-trace-hash", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=10240)
    args = parser.parse_args()

    specs = load_variant_specs(args.contract)
    if args.variant not in specs:
        raise ValueError(f"unknown variant: {args.variant}")
    probability, seed = specs[args.variant]
    trace_ids: list[str] = []
    correct = wrong = wrong_zero = wrong_tokens = 0
    with args.source_unmasked.open(encoding="utf-8") as source_handle, args.data.open(encoding="utf-8") as data_handle:
        source_lines = (line for line in source_handle if line.strip())
        data_lines = (line for line in data_handle if line.strip())
        for row_index, pair in enumerate(zip_longest(source_lines, data_lines), start=1):
            source_line, data_line = pair
            if source_line is None or data_line is None:
                raise ValueError(f"source/output row-count mismatch at {row_index}")
            source = json.loads(source_line)
            row = json.loads(data_line)
            source_metadata = source["metadata"]
            metadata = row["metadata"]
            trace_id = str(metadata["trace_id"])
            trace_ids.append(trace_id)
            if row["input_ids"] != source["input_ids"]:
                raise ValueError(f"input IDs changed: {trace_id}")
            source_without_weights = normalized_source_metadata(source_metadata)
            output_without_weights = normalized_source_metadata(metadata)
            source_weights = source_without_weights.pop("loss_weights")
            output_weights = output_without_weights.pop("loss_weights")
            if output_without_weights != source_without_weights:
                raise ValueError(f"source metadata changed: {trace_id}")
            assistant_count = int(metadata["assistant_token_count"])
            response_length = int(metadata["response_length"])
            if not 0 < assistant_count < response_length <= len(row["input_ids"]):
                raise ValueError(f"invalid response boundary: {trace_id}")
            if len(output_weights) != response_length or len(source_weights) != response_length:
                raise ValueError(f"loss-weight length drift: {trace_id}")
            if len(row["input_ids"]) > args.max_sequence_length:
                raise ValueError(f"sequence too long: {trace_id}")
            if metadata["variant"] != args.variant:
                raise ValueError(f"variant metadata drift: {trace_id}")
            if metadata["mask_seed"] != seed or metadata["mask_probability"] != probability:
                raise ValueError(f"mask metadata drift: {trace_id}")
            expected = [float(value) for value in source_weights]
            if bool(metadata["is_correct"]):
                correct += 1
            else:
                wrong += 1
                protected = [int(index) for index in metadata["protected_assistant_positions"]]
                expected[:assistant_count] = wrong_assistant_weights(
                    assistant_count,
                    protected,
                    seed,
                    probability,
                    trace_id,
                )
                protected_set = set(protected)
                ordinary = [
                    value for index, value in enumerate(expected[:assistant_count]) if index not in protected_set
                ]
                wrong_tokens += len(ordinary)
                wrong_zero += sum(value == 0.0 for value in ordinary)
            if [float(value) for value in output_weights] != expected:
                raise ValueError(f"loss weights differ from deterministic policy: {trace_id}")
    if (len(trace_ids), correct, wrong) != (2000, 1000, 1000):
        raise ValueError(f"class-count drift: {(len(trace_ids), correct, wrong)}")
    trace_hash = trace_set_sha256(trace_ids)
    if trace_hash != args.expected_trace_hash:
        raise ValueError(f"ordered trace-set drift: {trace_hash}")
    if wrong_zero == 0 or wrong_tokens == 0:
        raise ValueError("random mask did not affect ordinary wrong tokens")
    print(
        f"RANDOM_MASK_DATASET_VALID variant={args.variant} rows={len(trace_ids)} "
        f"masked={wrong_zero} trace_sha256={trace_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
