#!/usr/bin/env python3
"""Fail-closed structural validation for a pre-tokenized SFT variant."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse dataset expectations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--variant", choices=["correct_only", "weighted_tau_0p20", "unmasked"], required=True)
    parser.add_argument("--expected-rows", type=int, default=40000)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument("--im-end-id", type=int, default=151645)
    parser.add_argument("--eos-id", type=int, default=151643)
    return parser.parse_args()


def main() -> None:
    """Validate counts, exact terminals, alignment, and variant weights."""
    args = parse_args()
    rows = correct = wrong = 0
    trace_ids: set[str] = set()
    fractional_wrong_tokens = 0
    with Path(args.data).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            tokens = [int(value) for value in row["input_ids"]]
            metadata = row["metadata"]
            trace_id = str(metadata["trace_id"])
            if trace_id in trace_ids:
                raise ValueError(f"duplicate trace ID {trace_id}")
            trace_ids.add(trace_id)
            response_length = int(metadata["response_length"])
            assistant_count = int(metadata["assistant_token_count"])
            weights = [float(value) for value in metadata["loss_weights"]]
            if not 0 < len(tokens) <= args.max_sequence_length:
                raise ValueError(f"invalid sequence length for {trace_id}")
            if not 0 < assistant_count <= response_length <= len(tokens):
                raise ValueError(f"invalid response layout for {trace_id}")
            if len(weights) != response_length:
                raise ValueError(f"response-weight mismatch for {trace_id}")
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in weights):
                raise ValueError(f"invalid weight for {trace_id}")
            response_tokens = tokens[-response_length:]
            if args.eos_id in tokens:
                raise ValueError(f"synthetic or embedded endoftext in {trace_id}")
            im_end_positions = [index for index, token in enumerate(response_tokens) if token == args.im_end_id]
            if len(im_end_positions) != 1 or im_end_positions[0] < assistant_count:
                raise ValueError(f"invalid im_end placement for {trace_id}: {im_end_positions}")
            if weights[im_end_positions[0]] != 1.0:
                raise ValueError(f"im_end is not supervised for {trace_id}")
            assistant_weights = weights[:assistant_count]
            if any(value <= 0.0 for value in assistant_weights):
                raise ValueError(f"assistant token, including think tags, has zero weight for {trace_id}")
            is_correct = bool(metadata["is_correct"])
            if is_correct:
                correct += 1
                if any(value != 1.0 for value in assistant_weights):
                    raise ValueError(f"correct assistant token is not weight 1 for {trace_id}")
            else:
                wrong += 1
                fractional_wrong_tokens += sum(value < 1.0 for value in assistant_weights)
                if args.variant != "weighted_tau_0p20" and any(value != 1.0 for value in assistant_weights):
                    raise ValueError(f"unmasked wrong assistant token is fractional for {trace_id}")

    if rows != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {rows}")
    expected = (args.expected_rows, 0) if args.variant == "correct_only" else (args.expected_rows // 2, args.expected_rows // 2)
    if (correct, wrong) != expected:
        raise ValueError(f"class count mismatch correct={correct} wrong={wrong} expected={expected}")
    if args.variant == "weighted_tau_0p20" and fractional_wrong_tokens == 0:
        raise ValueError("weighted variant has no downweighted wrong token")
    print(
        f"DATASET_VALID variant={args.variant} rows={rows} correct={correct} wrong={wrong} "
        f"fractional_wrong_tokens={fractional_wrong_tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()
