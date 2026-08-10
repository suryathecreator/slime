#!/usr/bin/env python3
"""Fail closed on correct-only token boundaries, weights, and trace identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.qwen_correct_only_openr1_math220k_sft_2k.experiment import (
    load_contract,
    ordered_trace_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract()
    if args.run_key not in contract["runs"]:
        raise ValueError(f"unknown run key: {args.run_key}")
    run = contract["runs"][args.run_key]
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    expected_hash = stats["selection"][run["selection"]]["ordered_trace_sha256"]
    trace_ids: list[str] = []
    sequence_lengths: list[int] = []
    assistant_lengths: list[int] = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = [int(token) for token in row["input_ids"]]
            metadata = row["metadata"]
            trace_id = str(metadata["trace_id"])
            response_length = int(metadata["response_length"])
            assistant_count = int(metadata["assistant_token_count"])
            weights = [float(value) for value in metadata["loss_weights"]]
            if not 0 < len(tokens) <= int(run["max_sequence_length"]):
                raise ValueError(f"sequence length drift for {trace_id}")
            if not 0 < assistant_count <= int(run["trace_token_cap"]):
                raise ValueError(f"assistant length drift for {trace_id}")
            if response_length != assistant_count + 2 or len(weights) != response_length:
                raise ValueError(f"response boundary drift for {trace_id}")
            response = tokens[-response_length:]
            if response[-2:] != [151645, 198] or weights[-2:] != [1.0, 0.0]:
                raise ValueError(f"terminal policy drift for {trace_id}")
            if any(value != 1.0 for value in weights[:assistant_count]):
                raise ValueError(f"masked correct assistant token for {trace_id}")
            if any(token in (151643, 151644, 151645) for token in response[:assistant_count]):
                raise ValueError(f"assistant chat boundary for {trace_id}")
            if metadata.get("is_correct") is not True:
                raise ValueError(f"non-correct row in correct-only dataset: {trace_id}")
            trace_ids.append(trace_id)
            sequence_lengths.append(len(tokens))
            assistant_lengths.append(assistant_count)
    if len(trace_ids) != 2000 or len(set(trace_ids)) != 2000:
        raise ValueError(f"expected 2,000 unique traces, found {len(trace_ids)}/{len(set(trace_ids))}")
    actual_hash = ordered_trace_sha256(trace_ids)
    if actual_hash != expected_hash:
        raise ValueError(f"ordered trace-set mismatch: {actual_hash} != {expected_hash}")
    print(
        f"CORRECT_ONLY_DATASET_VALID run={args.run_key} rows=2000 "
        f"trace_sha256={actual_hash} min_seq={min(sequence_lengths)} max_seq={max(sequence_lengths)} "
        f"min_assistant={min(assistant_lengths)} max_assistant={max(assistant_lengths)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
