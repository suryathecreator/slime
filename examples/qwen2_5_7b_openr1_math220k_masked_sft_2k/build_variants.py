#!/usr/bin/env python3
"""Build nine masked Qwen2.5 variants while bypassing protected positions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    atomic_json,
    atomic_jsonl,
    sha256_file,
    stable_unit_interval,
    trace_set_sha256,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.build_weighted import inverse_margin_weight
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.score_margins import record_path


VARIANTS = (
    "random_mask_25", "random_mask_50", "random_mask_70",
    "random_mask_80", "random_mask_90", "inverse_tau_0p20",
    "inverse_tau_0p05", "margin_mask", "prob_ratio_mask",
)
RANDOM_PROBABILITIES = {
    "random_mask_25": 0.25, "random_mask_50": 0.50, "random_mask_70": 0.70,
    "random_mask_80": 0.80, "random_mask_90": 0.90,
}
INVERSE_TAUS = {"inverse_tau_0p20": 0.20, "inverse_tau_0p05": 0.05}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unmasked-data", required=True)
    parser.add_argument("--margin-dir", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--expected-rows", type=int, default=2000)
    parser.add_argument("--expected-correct", type=int, default=1000)
    parser.add_argument("--expected-wrong", type=int, default=1000)
    parser.add_argument("--output", action="append", required=True)
    return parser.parse_args()


def output_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        variant, separator, path = value.partition("=")
        if separator != "=" or variant not in VARIANTS or not path or variant in result:
            raise ValueError(f"invalid --output value: {value}")
        result[variant] = path
    if set(result) != set(VARIANTS):
        raise ValueError(f"variant output mismatch: {sorted(result)}")
    return result


def load_margins(margin_dir: str, metadata: dict[str, Any]) -> list[float]:
    path = record_path(margin_dir, int(metadata["wrong_index"]))
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ("trace_id", "assistant_token_sha256", "wrong_index"):
        if record.get(key) != metadata.get(key):
            raise ValueError(f"margin alignment mismatch for {key}: {path}")
    margins = record.get("delta_logprobs")
    if not isinstance(margins, list) or len(margins) != int(metadata["assistant_token_count"]):
        raise ValueError(f"margin length mismatch: {path}")
    values = [float(value) for value in margins]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"non-finite margin: {path}")
    return values


def wrong_weights(
    variant: str,
    margins: list[float],
    trace_id: str,
    seed: int,
    protected_positions: list[int],
) -> list[float]:
    """Compute weights only on ordinary tokens; controls never enter the policy."""
    protected = {int(index) for index in protected_positions}
    if any(index < 0 or index >= len(margins) for index in protected):
        raise ValueError("protected assistant index outside margin sequence")
    values = [1.0] * len(margins)
    for index, margin in enumerate(margins):
        if index in protected:
            continue
        if variant in RANDOM_PROBABILITIES:
            values[index] = float(
                stable_unit_interval(seed, trace_id, index) >= RANDOM_PROBABILITIES[variant]
            )
        elif variant in INVERSE_TAUS:
            values[index] = inverse_margin_weight(margin, INVERSE_TAUS[variant])
        elif variant == "margin_mask":
            values[index] = float(margin > 0.0)
        elif variant == "prob_ratio_mask":
            mask_probability = 1.0 if margin <= 0.0 else math.exp(-margin)
            values[index] = float(stable_unit_interval(seed, trace_id, index) >= mask_probability)
        else:
            raise ValueError(f"unknown variant: {variant}")
    return values


def variant_rows(args: argparse.Namespace, variant: str, aggregate: dict[str, Any]) -> Iterator[dict[str, Any]]:
    trace_ids: list[str] = []
    with Path(args.unmasked_data).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row["metadata"]
            trace_id = str(metadata["trace_id"])
            trace_ids.append(trace_id)
            assistant_count = int(metadata["assistant_token_count"])
            weights = [float(value) for value in metadata["loss_weights"]]
            protected = [int(index) for index in metadata["protected_assistant_positions"]]
            if not protected or any(weights[index] != 1.0 for index in protected):
                raise ValueError(f"invalid protected-token baseline: {trace_id}")
            aggregate["rows"] += 1
            if bool(metadata["is_correct"]):
                aggregate["correct_rows"] += 1
            else:
                aggregate["wrong_rows"] += 1
                margins = load_margins(args.margin_dir, metadata)
                assistant_weights = wrong_weights(
                    variant, margins, trace_id, args.mask_seed, protected
                )
                if any(assistant_weights[index] != 1.0 for index in protected):
                    raise AssertionError(f"protected token changed: {trace_id}")
                weights[:assistant_count] = assistant_weights
                aggregate["wrong_tokens"] += assistant_count - len(protected)
                aggregate["wrong_weight_sum"] += sum(
                    value for index, value in enumerate(assistant_weights) if index not in set(protected)
                )
                aggregate["wrong_zero_tokens"] += sum(
                    value == 0.0 for index, value in enumerate(assistant_weights) if index not in set(protected)
                )
                aggregate["wrong_fractional_tokens"] += sum(
                    0.0 < value < 1.0 for index, value in enumerate(assistant_weights) if index not in set(protected)
                )
                aggregate["protected_tokens"] += len(protected)
            metadata["loss_weights"] = weights
            metadata["loss_weight_policy"] = f"{variant};all_protected_positions_bypass_policy"
            metadata["mask_seed"] = args.mask_seed
            metadata["variant"] = variant
            yield row
    aggregate["ordered_trace_ids_sha256"] = trace_set_sha256(trace_ids)


def main() -> None:
    args = parse_args()
    outputs = output_map(args.output)
    for destination in [*outputs.values(), args.stats_output]:
        if Path(destination).exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
    all_stats: dict[str, Any] = {
        "mask_seed": args.mask_seed,
        "protected_policy": "bypass all random draws, comparisons, and transforms; exact weight 1",
        "variants": {},
    }
    expected_trace_hash: str | None = None
    for variant in VARIANTS:
        aggregate = {
            "rows": 0, "correct_rows": 0, "wrong_rows": 0, "wrong_tokens": 0,
            "wrong_weight_sum": 0.0, "wrong_zero_tokens": 0,
            "wrong_fractional_tokens": 0, "protected_tokens": 0,
        }
        atomic_jsonl(outputs[variant], variant_rows(args, variant, aggregate))
        if (aggregate["rows"], aggregate["correct_rows"], aggregate["wrong_rows"]) != (
            args.expected_rows, args.expected_correct, args.expected_wrong
        ):
            raise ValueError(f"row-count mismatch for {variant}: {aggregate}")
        trace_hash = str(aggregate["ordered_trace_ids_sha256"])
        if expected_trace_hash is None:
            expected_trace_hash = trace_hash
        elif trace_hash != expected_trace_hash:
            raise ValueError(f"trace-set drift for {variant}")
        aggregate["wrong_mean_weight"] = aggregate["wrong_weight_sum"] / aggregate["wrong_tokens"]
        aggregate["dataset_sha256"] = sha256_file(outputs[variant])
        all_stats["variants"][variant] = aggregate
        print(f"VARIANT_BUILD_COMPLETE variant={variant}", flush=True)
    all_stats["ordered_mixed_trace_ids_sha256"] = expected_trace_hash
    atomic_json(args.stats_output, all_stats)


if __name__ == "__main__":
    main()
