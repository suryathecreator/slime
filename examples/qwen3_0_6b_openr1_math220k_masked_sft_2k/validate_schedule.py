#!/usr/bin/env python3
"""Prove the realized 2K dynamic schedule has one microbatch per DP rank."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import atomic_json
from slime.utils.dp_schedule import build_dp_schedule


def lengths(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as handle:
        return [len(json.loads(line)["input_ids"]) for line in handle if line.strip()]


def audit(path: Path, seed: int, batch_size: int, dp_size: int, max_tokens: int) -> dict:
    values = lengths(path)
    permutation = list(range(len(values)))
    random.Random(seed).shuffle(permutation)
    shuffled = [values[index] for index in permutation]
    args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=max_tokens,
        micro_batch_size=1,
        balance_by_flops=False,
        balance_data=False,
    )
    partitions, micro_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        args,
        {
            "dp_size": dp_size,
            "cp_size": 1,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        },
        shuffled,
        global_batch_size=batch_size,
        rollout_indices=list(range(len(shuffled))),
    )
    expected_steps = len(values) // batch_size
    if len(values) != 2000 or expected_steps != 125:
        raise ValueError(f"unexpected dataset/update count for {path}: {len(values)}/{expected_steps}")
    if num_microbatches != [1] * expected_steps:
        bad = [index for index, value in enumerate(num_microbatches) if value != 1]
        raise ValueError(f"effective gradient accumulation is not one for {path}: {bad[:10]}")
    if global_batch_sizes != [batch_size] * expected_steps:
        raise ValueError(f"global batch-size drift for {path}")
    if sum(map(len, partitions)) != len(values):
        raise ValueError(f"schedule does not cover every row in {path}")
    if any(len(rank_mbs) != expected_steps for rank_mbs in micro_indices):
        raise ValueError(f"per-rank microbatch-count drift for {path}")
    return {
        "dataset": str(path),
        "rows": len(values),
        "optimizer_updates": expected_steps,
        "effective_gradient_accumulation_steps": 1,
        "min_sequence_length": min(values),
        "max_sequence_length": max(values),
        "mean_sequence_length": sum(values) / len(values),
        "rows_per_rank": [len(partition) for partition in partitions],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=32768)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = {
        "correct_only": audit(
            args.correct, args.seed, args.global_batch_size, args.dp_size, args.max_tokens_per_gpu
        ),
        "mixed_all_variants": audit(
            args.mixed, args.seed, args.global_batch_size, args.dp_size, args.max_tokens_per_gpu
        ),
        "contract": {
            "global_batch_size": args.global_batch_size,
            "data_parallel_size": args.dp_size,
            "max_tokens_per_gpu": args.max_tokens_per_gpu,
            "max_sequence_length": 32768,
        },
    }
    atomic_json(args.output, result)
    print("DYNAMIC_SCHEDULE_VALID updates=125 grad_accum=1 dp=4", flush=True)


if __name__ == "__main__":
    main()
