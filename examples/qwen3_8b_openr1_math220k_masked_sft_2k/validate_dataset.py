#!/usr/bin/env python3
"""Fail-closed validation of 2K trace reuse and loss-mask semantics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from examples.qwen3_8b_openr1_math220k_masked_sft_2k.build_variants import (
    INVERSE_TAUS,
    RANDOM_PROBABILITIES,
    VARIANTS,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_2k.data_utils import (
    trace_set_sha256,
)


ALL_VARIANTS = ("correct_only", "unmasked", *VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--variant", choices=ALL_VARIANTS, required=True)
    parser.add_argument("--expected-trace-hash")
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument("--im-end-id", type=int, default=151645)
    parser.add_argument("--eos-id", type=int, default=151643)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            if not 0 < assistant_count < response_length <= len(tokens):
                raise ValueError(f"invalid response boundary: {trace_id}")
            if len(weights) != response_length:
                raise ValueError(f"response-weight mismatch: {trace_id}")
            if any(
                not math.isfinite(value) or value < 0.0 or value > 1.0
                for value in weights
            ):
                raise ValueError(f"invalid loss weight: {trace_id}")
            response_ids = tokens[-response_length:]
            if args.eos_id in tokens:
                raise ValueError(f"synthetic or embedded endoftext: {trace_id}")
            im_end_positions = [
                index
                for index, token in enumerate(response_ids)
                if token == args.im_end_id
            ]
            if im_end_positions != [assistant_count]:
                raise ValueError(
                    f"im_end is not the exact assistant terminal: {trace_id} "
                    f"{im_end_positions}"
                )
            if weights[assistant_count] != 1.0:
                raise ValueError(f"im_end is not supervised: {trace_id}")
            if any(value != 0.0 for value in weights[assistant_count + 1 :]):
                raise ValueError(f"template suffix is supervised: {trace_id}")
            protected = [
                int(index)
                for index in metadata.get("protected_assistant_positions", [])
            ]
            if not protected or any(
                index < 0
                or index >= assistant_count
                or weights[index] != 1.0
                for index in protected
            ):
                raise ValueError(f"think tags are not fully supervised: {trace_id}")
            assistant_weights = weights[:assistant_count]
            is_correct = bool(metadata["is_correct"])
            if is_correct:
                correct += 1
                if any(value != 1.0 for value in assistant_weights):
                    raise ValueError(f"correct token is masked: {trace_id}")
            else:
                wrong += 1
                zero_wrong += sum(value == 0.0 for value in assistant_weights)
                fractional_wrong += sum(
                    0.0 < value < 1.0 for value in assistant_weights
                )
                if args.variant == "unmasked" and any(
                    value != 1.0 for value in assistant_weights
                ):
                    raise ValueError(f"unmasked wrong token changed: {trace_id}")
                if args.variant in (*RANDOM_PROBABILITIES, "margin_mask", "prob_ratio_mask"):
                    if any(value not in (0.0, 1.0) for value in assistant_weights):
                        raise ValueError(f"binary mask is fractional: {trace_id}")
                if args.variant in INVERSE_TAUS and any(
                    not 0.0 < value <= 1.0 for value in assistant_weights
                ):
                    raise ValueError(f"inverse weight is outside (0,1]: {trace_id}")

    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("duplicate trace IDs")
    expected_counts = (
        (2000, 0) if args.variant == "correct_only" else (1000, 1000)
    )
    if (correct, wrong) != expected_counts:
        raise ValueError(
            f"class-count mismatch: correct={correct} wrong={wrong} "
            f"expected={expected_counts}"
        )
    if args.variant in (*RANDOM_PROBABILITIES, "margin_mask", "prob_ratio_mask"):
        if zero_wrong == 0:
            raise ValueError(f"binary variant masks no wrong tokens: {args.variant}")
    if args.variant in INVERSE_TAUS and fractional_wrong == 0:
        raise ValueError(f"inverse variant has no fractional weights: {args.variant}")
    trace_hash = trace_set_sha256(trace_ids)
    if args.expected_trace_hash and trace_hash != args.expected_trace_hash:
        raise ValueError(
            f"ordered trace-set mismatch: {trace_hash} != {args.expected_trace_hash}"
        )
    print(
        f"DATASET_2K_VALID variant={args.variant} rows={len(trace_ids)} "
        f"correct={correct} wrong={wrong} zero_wrong={zero_wrong} "
        f"fractional_wrong={fractional_wrong} trace_sha256={trace_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
