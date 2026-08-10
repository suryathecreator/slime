#!/usr/bin/env python3
"""Fail closed on Qwen2.5 data boundaries and protected supervision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants import (
    INVERSE_TAUS,
    RANDOM_PROBABILITIES,
    VARIANTS,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import trace_set_sha256

ALL_VARIANTS = ("correct_only", "unmasked", *VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--variant", choices=ALL_VARIANTS, required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-trace-hash")
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    added_ids = {int(value) for value in inventory["protected_added_token_ids"]}
    trace_ids: list[str] = []
    correct = wrong = zero_wrong = fractional_wrong = 0
    with Path(args.data).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = [int(value) for value in row["input_ids"]]
            metadata = row["metadata"]
            trace_id = str(metadata["trace_id"])
            trace_ids.append(trace_id)
            response_length = int(metadata["response_length"])
            assistant_count = int(metadata["assistant_token_count"])
            weights = [float(value) for value in metadata["loss_weights"]]
            if not 0 < len(tokens) <= args.max_sequence_length:
                raise ValueError(f"invalid sequence length: {trace_id}")
            if not 0 < assistant_count < response_length <= len(tokens) or len(weights) != response_length:
                raise ValueError(f"invalid response boundary: {trace_id}")
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights):
                raise ValueError(f"invalid loss weight: {trace_id}")
            response_ids = tokens[-response_length:]
            assistant_ids = response_ids[:assistant_count]
            if any(token in assistant_ids for token in (151643, 151644, 151645)):
                raise ValueError(f"embedded chat boundary: {trace_id}")
            if [index for index, token in enumerate(response_ids) if token == 151645] != [assistant_count]:
                raise ValueError(f"im_end is not the exact assistant terminal: {trace_id}")
            if 151643 in tokens or weights[assistant_count] != 1.0:
                raise ValueError(f"EOS/im_end policy violation: {trace_id}")
            if any(value != 0.0 for value in weights[assistant_count + 1:]):
                raise ValueError(f"template suffix is supervised: {trace_id}")
            protected = {int(index) for index in metadata.get("protected_assistant_positions", [])}
            reasons = metadata.get("protected_token_reasons", {})
            expected_added = {index for index, token in enumerate(assistant_ids) if token in added_ids}
            literal_positions = {
                str(literal): {int(index) for index in positions}
                for literal, positions in metadata.get("protected_literal_positions", {}).items()
            }
            if set(literal_positions) != {"<think>", "</think>"}:
                raise ValueError(f"literal protection metadata drift: {trace_id}")
            expected_think = set().union(*literal_positions.values())
            expected = expected_added | expected_think
            if not literal_positions["<think>"] or protected != expected:
                raise ValueError(f"protected-position inventory drift: {trace_id}")
            if (
                set(map(int, reasons)) != protected
                or any(weights[index] != 1.0 for index in protected)
                or any(
                    f"literal_markup:{literal}" not in reasons[str(index)]
                    for literal, positions in literal_positions.items()
                    for index in positions
                )
            ):
                raise ValueError(f"protected token not exactly supervised: {trace_id}")
            assistant_weights = weights[:assistant_count]
            if bool(metadata["is_correct"]):
                correct += 1
                if any(value != 1.0 for value in assistant_weights):
                    raise ValueError(f"correct token is masked: {trace_id}")
            else:
                wrong += 1
                ordinary = [
                    value for index, value in enumerate(assistant_weights) if index not in protected
                ]
                zero_wrong += sum(value == 0.0 for value in ordinary)
                fractional_wrong += sum(0.0 < value < 1.0 for value in ordinary)
                if args.variant == "unmasked" and any(value != 1.0 for value in ordinary):
                    raise ValueError(f"unmasked wrong token changed: {trace_id}")
                if args.variant in (*RANDOM_PROBABILITIES, "margin_mask", "prob_ratio_mask") and any(
                    value not in (0.0, 1.0) for value in ordinary
                ):
                    raise ValueError(f"binary mask is fractional: {trace_id}")
                if args.variant in INVERSE_TAUS and any(not 0.0 < value <= 1.0 for value in ordinary):
                    raise ValueError(f"inverse weight outside (0,1]: {trace_id}")
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("duplicate trace IDs")
    expected_counts = (2000, 0) if args.variant == "correct_only" else (1000, 1000)
    if (correct, wrong) != expected_counts:
        raise ValueError(f"class-count mismatch: {(correct, wrong)} != {expected_counts}")
    if args.variant in (*RANDOM_PROBABILITIES, "margin_mask", "prob_ratio_mask") and zero_wrong == 0:
        raise ValueError(f"binary variant masks no ordinary wrong tokens: {args.variant}")
    if args.variant in INVERSE_TAUS and fractional_wrong == 0:
        raise ValueError(f"inverse variant has no fractional ordinary weights: {args.variant}")
    trace_hash = trace_set_sha256(trace_ids)
    if args.expected_trace_hash and trace_hash != args.expected_trace_hash:
        raise ValueError(f"ordered trace-set mismatch: {trace_hash}")
    print(
        f"DATASET_QWEN25_2K_VALID variant={args.variant} rows={len(trace_ids)} "
        f"correct={correct} wrong={wrong} trace_sha256={trace_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
