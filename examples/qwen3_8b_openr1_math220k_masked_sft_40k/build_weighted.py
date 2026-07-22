#!/usr/bin/env python3
"""Merge margin scores and build the tau=0.20 weighted full-SFT dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import atomic_json, atomic_jsonl, sha256_file
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.score_margins import record_path


def parse_args() -> argparse.Namespace:
    """Parse weighted-dataset build arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unmasked-data", required=True)
    parser.add_argument("--margin-dir", required=True)
    parser.add_argument("--merged-margins", required=True)
    parser.add_argument("--weighted-output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--expected-rows", type=int, default=40000)
    parser.add_argument("--expected-correct", type=int, default=20000)
    parser.add_argument("--expected-wrong", type=int, default=20000)
    return parser.parse_args()


def inverse_margin_weight(margin: float, tau: float) -> float:
    """Apply the Axolotl inverse-margin continuous weighting rule."""
    if tau <= 0.0 or not math.isfinite(margin):
        raise ValueError("tau must be positive and margin finite")
    return 1.0 if margin >= 0.0 else 1.0 / (1.0 + (-margin / tau))


def load_score(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Load and align one atomic margin record with a training row."""
    if not path.is_file():
        raise FileNotFoundError(f"missing margin record {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ("trace_id", "assistant_token_sha256", "wrong_index"):
        if record.get(key) != metadata.get(key):
            raise ValueError(f"margin alignment mismatch for {key}: {path}")
    margins = record.get("delta_logprobs")
    if not isinstance(margins, list) or len(margins) != int(metadata["assistant_token_count"]):
        raise ValueError(f"margin length mismatch: {path}")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in margins):
        raise ValueError(f"non-finite margin: {path}")
    return record


def margin_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """Yield the complete margin set in stable wrong-index order."""
    for wrong_index in range(args.expected_wrong):
        path = record_path(args.margin_dir, wrong_index)
        if not path.is_file():
            raise FileNotFoundError(f"missing margin record {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("wrong_index", -1)) != wrong_index:
            raise ValueError(f"wrong-index mismatch in {path}")
        yield record


def weighted_rows(args: argparse.Namespace, aggregate: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Stream weighted rows while checking exact unmasked row identity."""
    row_count = 0
    correct_count = 0
    wrong_count = 0
    seen_trace_ids: set[str] = set()
    with Path(args.unmasked_data).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            metadata = row["metadata"]
            trace_id = str(metadata["trace_id"])
            if trace_id in seen_trace_ids:
                raise ValueError(f"duplicate training trace {trace_id}")
            seen_trace_ids.add(trace_id)
            assistant_count = int(metadata["assistant_token_count"])
            weights = [float(value) for value in metadata["loss_weights"]]
            if len(weights) != int(metadata["response_length"]):
                raise ValueError(f"unmasked response-weight mismatch for {trace_id}")
            if any(abs(value - 1.0) > 1e-12 for value in weights[:assistant_count]):
                raise ValueError(f"unmasked assistant token is not weight 1 for {trace_id}")
            if bool(metadata["is_correct"]):
                correct_count += 1
            else:
                wrong_count += 1
                wrong_index = int(metadata["wrong_index"])
                record = load_score(record_path(args.margin_dir, wrong_index), metadata)
                assistant_weights = [inverse_margin_weight(float(value), args.tau) for value in record["delta_logprobs"]]
                if any(not 0.0 < value <= 1.0 for value in assistant_weights):
                    raise ValueError(f"wrong-token weight outside (0,1] for {trace_id}")
                weights[:assistant_count] = assistant_weights
                for value in assistant_weights:
                    aggregate["wrong_token_count"] += 1
                    aggregate["wrong_weight_sum"] += value
                    aggregate["wrong_weight_min"] = min(aggregate["wrong_weight_min"], value)
                    aggregate["wrong_weight_max"] = max(aggregate["wrong_weight_max"], value)
                    aggregate["wrong_fraction_below_0p1_count"] += int(value < 0.1)
                    aggregate["wrong_fraction_below_0p5_count"] += int(value < 0.5)
                    aggregate["wrong_fraction_equal_1_count"] += int(abs(value - 1.0) < 1e-12)
            # The final response token(s) retain the prebuilt template policy:
            # exactly im_end is supervised; trailing template whitespace is not.
            metadata["loss_weights"] = weights
            metadata["loss_weight_policy"] = (
                "correct=1;wrong=max_margin_or_inverse_decay;im_end=1;template_newline=0"
            )
            metadata["weight_tau"] = args.tau
            yield row

    if row_count != args.expected_rows or correct_count != args.expected_correct or wrong_count != args.expected_wrong:
        raise ValueError(
            f"weighted row counts mismatch rows={row_count} correct={correct_count} wrong={wrong_count}"
        )
    aggregate.update({"rows": row_count, "correct_rows": correct_count, "wrong_rows": wrong_count})


def main() -> None:
    """Build merged margins, weighted data, and fail-closed audit stats."""
    args = parse_args()
    for path in (args.merged_margins, args.weighted_output, args.stats_output):
        if Path(path).exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    atomic_jsonl(args.merged_margins, margin_rows(args))
    aggregate: dict[str, Any] = {
        "wrong_fraction_below_0p1_count": 0,
        "wrong_fraction_below_0p5_count": 0,
        "wrong_fraction_equal_1_count": 0,
        "wrong_token_count": 0,
        "wrong_weight_max": float("-inf"),
        "wrong_weight_min": float("inf"),
        "wrong_weight_sum": 0.0,
    }
    atomic_jsonl(args.weighted_output, weighted_rows(args, aggregate))
    token_count = int(aggregate["wrong_token_count"])
    if token_count <= 0:
        raise ValueError("weighted dataset has no wrong tokens")
    stats = {
        "formula": "weight=1 if margin>=0 else 1/(1+(-margin/tau))",
        "margin_definition": "logp_answer_conditioned_minus_logp_unconditioned",
        "normalization": "sum_weighted_token_losses_divided_by_sum_loss_weights",
        "tau": args.tau,
        "terminal_policy": "assistant tokens and im_end supervised; no synthetic endoftext; trailing newline weight 0",
        "rows": aggregate["rows"],
        "correct_rows": aggregate["correct_rows"],
        "wrong_rows": aggregate["wrong_rows"],
        "wrong_tokens": {
            "count": token_count,
            "weight_sum": aggregate["wrong_weight_sum"],
            "mean": aggregate["wrong_weight_sum"] / token_count,
            "min": aggregate["wrong_weight_min"],
            "max": aggregate["wrong_weight_max"],
            "fraction_below_0p1": aggregate["wrong_fraction_below_0p1_count"] / token_count,
            "fraction_below_0p5": aggregate["wrong_fraction_below_0p5_count"] / token_count,
            "fraction_equal_1": aggregate["wrong_fraction_equal_1_count"] / token_count,
        },
        "artifacts": {
            "merged_margins": {"path": args.merged_margins, "sha256": sha256_file(args.merged_margins)},
            "weighted_dataset": {"path": args.weighted_output, "sha256": sha256_file(args.weighted_output)},
        },
    }
    atomic_json(args.stats_output, stats)
    print(f"WEIGHTED_BUILD_COMPLETE stats={args.stats_output}", flush=True)


if __name__ == "__main__":
    main()
