#!/usr/bin/env python3
"""Build deterministic low-rate random masks from the frozen unmasked data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    atomic_json,
    atomic_jsonl,
    sha256_file,
    stable_unit_interval,
    trace_set_sha256,
)

EXPECTED_VARIANTS = (
    "random_mask_05_seed42",
    "random_mask_15_seed42",
    "random_mask_05_seed43",
    "random_mask_15_seed43",
    "random_mask_05_seed44",
    "random_mask_15_seed44",
)


def load_variant_specs(contract_path: Path) -> dict[str, tuple[float, int]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    specs = {str(item["id"]): (float(item["probability"]), int(item["seed"])) for item in contract["variants"]}
    if tuple(specs) != EXPECTED_VARIANTS:
        raise ValueError("variant order drift")
    if set(specs.values()) != {
        (0.05, 42),
        (0.15, 42),
        (0.05, 43),
        (0.15, 43),
        (0.05, 44),
        (0.15, 44),
    }:
        raise ValueError("mask probability or seed drift")
    return specs


def wrong_assistant_weights(
    assistant_count: int,
    protected_positions: list[int],
    seed: int,
    probability: float,
    trace_id: str,
) -> list[float]:
    protected = {int(index) for index in protected_positions}
    if any(index < 0 or index >= assistant_count for index in protected):
        raise ValueError(f"protected position outside assistant: {trace_id}")
    return [
        1.0 if index in protected else float(stable_unit_interval(seed, trace_id, index) >= probability)
        for index in range(assistant_count)
    ]


def variant_rows(
    source_path: Path,
    variant: str,
    probability: float,
    seed: int,
    aggregate: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    trace_ids: list[str] = []
    with source_path.open(encoding="utf-8") as handle:
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
                raise ValueError(f"invalid source protected-token weights: {trace_id}")
            if any(value != 1.0 for value in weights[:assistant_count]):
                raise ValueError(f"source data is not unmasked: {trace_id}")
            aggregate["rows"] += 1
            if bool(metadata["is_correct"]):
                aggregate["correct_rows"] += 1
            else:
                aggregate["wrong_rows"] += 1
                assistant_weights = wrong_assistant_weights(
                    assistant_count,
                    protected,
                    seed,
                    probability,
                    trace_id,
                )
                weights[:assistant_count] = assistant_weights
                protected_set = set(protected)
                ordinary = [value for index, value in enumerate(assistant_weights) if index not in protected_set]
                aggregate["wrong_tokens"] += len(ordinary)
                aggregate["wrong_zero_tokens"] += sum(value == 0.0 for value in ordinary)
                aggregate["wrong_weight_sum"] += sum(ordinary)
                aggregate["protected_tokens"] += len(protected)
            metadata["loss_weights"] = weights
            metadata["loss_weight_policy"] = (
                f"random_mask_probability_{probability:.2f}_seed_{seed};" "all_protected_positions_bypass_policy"
            )
            metadata["mask_probability"] = probability
            metadata["mask_seed"] = seed
            metadata["variant"] = variant
            yield row
    aggregate["ordered_trace_ids_sha256"] = trace_set_sha256(trace_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-unmasked", type=Path, required=True)
    parser.add_argument("--source-prep-stats", type=Path, required=True)
    parser.add_argument("--source-tokenizer-inventory", type=Path, required=True)
    parser.add_argument("--source-schedule-audit", type=Path, required=True)
    parser.add_argument("--source-training-status", type=Path, required=True)
    parser.add_argument("--source-comparison-sources", type=Path, required=True)
    parser.add_argument("--source-artifacts-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    specs = load_variant_specs(args.contract)
    outputs = {variant: args.output_dir / f"{variant}_2000.jsonl" for variant in specs}
    for destination in (
        *outputs.values(),
        args.stats_output,
        args.source_artifacts_output,
    ):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
    source_files = {
        "comparison_sources": (
            args.source_comparison_sources,
            "comparison_sources_sha256",
        ),
        "schedule_audit": (args.source_schedule_audit, "schedule_audit_sha256"),
        "selection_and_tokenization_stats": (
            args.source_prep_stats,
            "selection_and_tokenization_stats_sha256",
        ),
        "tokenizer_control_inventory": (
            args.source_tokenizer_inventory,
            "tokenizer_control_inventory_sha256",
        ),
        "training_status": (args.source_training_status, "training_status_sha256"),
        "unmasked_dataset": (args.source_unmasked, "unmasked_dataset_sha256"),
    }
    source_records = {}
    for name, (path, hash_key) in source_files.items():
        actual = sha256_file(path)
        expected = contract["source_experiment"][hash_key]
        if actual != expected:
            raise ValueError(f"source artifact hash drift: {name}")
        source_records[name] = {"path": str(path.resolve()), "sha256": actual}
    atomic_json(
        args.source_artifacts_output,
        {
            "artifact_schema_version": 1,
            "ordered_mixed_trace_ids_sha256": contract["source_experiment"]["ordered_mixed_trace_ids_sha256"],
            "source_contract_hash": contract["source_experiment"]["contract_hash"],
            "source_files": source_records,
        },
    )
    expected_trace_hash: str | None = None
    all_stats: dict[str, Any] = {
        "artifact_schema_version": 1,
        "protected_policy": "all protected assistant positions remain exact weight 1",
        "source_unmasked_sha256": sha256_file(args.source_unmasked),
        "variants": {},
        "within_seed_nesting": True,
    }
    for variant, (probability, seed) in specs.items():
        aggregate: dict[str, Any] = {
            "correct_rows": 0,
            "protected_tokens": 0,
            "rows": 0,
            "wrong_rows": 0,
            "wrong_tokens": 0,
            "wrong_weight_sum": 0.0,
            "wrong_zero_tokens": 0,
        }
        atomic_jsonl(
            outputs[variant],
            variant_rows(
                args.source_unmasked,
                variant,
                probability,
                seed,
                aggregate,
            ),
        )
        counts = (
            aggregate["rows"],
            aggregate["correct_rows"],
            aggregate["wrong_rows"],
        )
        if counts != (2000, 1000, 1000):
            raise ValueError(f"row-count drift for {variant}: {counts}")
        trace_hash = str(aggregate["ordered_trace_ids_sha256"])
        if expected_trace_hash is None:
            expected_trace_hash = trace_hash
        elif trace_hash != expected_trace_hash:
            raise ValueError(f"ordered trace-set drift for {variant}")
        aggregate["dataset_sha256"] = sha256_file(outputs[variant])
        aggregate["mask_probability"] = probability
        aggregate["mask_seed"] = seed
        aggregate["wrong_masked_fraction"] = aggregate["wrong_zero_tokens"] / aggregate["wrong_tokens"]
        aggregate["wrong_mean_weight"] = aggregate["wrong_weight_sum"] / aggregate["wrong_tokens"]
        all_stats["variants"][variant] = aggregate
        print(
            f"RANDOM_MASK_BUILD_COMPLETE variant={variant} " f"masked={aggregate['wrong_zero_tokens']}",
            flush=True,
        )
    all_stats["ordered_mixed_trace_ids_sha256"] = expected_trace_hash
    atomic_json(args.stats_output, all_stats)


if __name__ == "__main__":
    main()
