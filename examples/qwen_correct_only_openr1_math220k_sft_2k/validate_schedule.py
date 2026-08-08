#!/usr/bin/env python3
"""Audit all four epochs of the realized dynamic microbatch schedule."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import atomic_json
from slime.utils.dp_schedule import build_dp_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--max-tokens-per-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with args.data.open(encoding="utf-8") as handle:
        one_epoch = [len(json.loads(line)["input_ids"]) for line in handle if line.strip()]
    if len(one_epoch) != 2000:
        raise ValueError(f"expected 2,000 rows, found {len(one_epoch)}")
    lengths: list[int] = []
    for epoch in range(4):
        epoch_lengths = list(one_epoch)
        random.Random(f"{args.seed}:epoch:{epoch}").shuffle(epoch_lengths)
        lengths.extend(epoch_lengths)
    schedule_args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        micro_batch_size=1,
        balance_by_flops=False,
        balance_data=False,
    )
    partitions, micro_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        schedule_args,
        {
            "dp_size": args.dp_size,
            "cp_size": 1,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        },
        lengths,
        global_batch_size=64,
        rollout_indices=list(range(len(lengths))),
    )
    if len(num_microbatches) != 125 or global_batch_sizes != [64] * 125:
        raise ValueError("optimizer-update/global-batch schedule drift")
    if sum(map(len, partitions)) != 8000:
        raise ValueError("dynamic schedule does not cover four epochs")
    if any(len(rank_batches) != 125 for rank_batches in micro_indices):
        raise ValueError("per-rank optimizer update count drift")
    histogram = {str(key): value for key, value in sorted(Counter(num_microbatches).items())}
    atomic_json(
        args.output,
        {
            "data_parallel_size": args.dp_size,
            "epochs": 4,
            "examples_per_global_update": 64,
            "global_examples": 8000,
            "max_tokens_per_gpu": args.max_tokens_per_gpu,
            "microbatches_per_update_histogram": histogram,
            "optimizer_updates": 125,
            "rows_per_rank": [len(partition) for partition in partitions],
        },
    )
    print(
        f"DYNAMIC_SCHEDULE_VALID updates=125 gbs=64 dp={args.dp_size} histogram={histogram}",
        flush=True,
    )


if __name__ == "__main__":
    main()
