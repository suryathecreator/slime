#!/usr/bin/env python3
"""Build the final base/correct-only/weighted/unmasked MATH-500 table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.qwen3_8b_32b_full_repro.math500_scorer import SCORER_VERSION
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import atomic_json, sha256_file


def parse_args() -> argparse.Namespace:
    """Parse summary paths and report destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--correct-only", required=True)
    parser.add_argument("--weighted", required=True)
    parser.add_argument("--unmasked", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def summary_row(label: str, path: str, base_accuracy: float) -> dict[str, Any]:
    """Extract the compact metrics table row from one evaluator summary."""
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(summary.get("per_problem", [])) != 500:
        raise ValueError(f"summary is not a complete MATH-500 evaluation: {path}")
    if summary.get("scorer_version") != SCORER_VERSION:
        raise ValueError(f"unexpected scorer for {path}: {summary.get('scorer_version')}")
    accuracy = float(summary["pass_at_1"])
    return {
        "accuracy": accuracy,
        "bootstrap_95_ci": [float(value) for value in summary["bootstrap_95_ci"]],
        "cap_hits": int(summary["cap_hits"]),
        "correct": int(summary["correct"]),
        "delta_vs_base": accuracy - base_accuracy,
        "label": label,
        "parseable": int(summary["parseable"]),
        "summary_path": path,
        "summary_sha256": sha256_file(path),
    }


def main() -> None:
    """Verify base provenance and write JSON plus Markdown reports."""
    args = parse_args()
    actual_base_hash = sha256_file(args.base)
    if actual_base_hash != args.base_sha256:
        raise ValueError(f"base summary hash mismatch: {actual_base_hash}")
    base_summary = json.loads(Path(args.base).read_text(encoding="utf-8"))
    base_accuracy = float(base_summary["pass_at_1"])
    rows = [
        summary_row("Base 8B", args.base, base_accuracy),
        summary_row("Post-SFT correct-only", args.correct_only, base_accuracy),
        summary_row("Post-SFT weighted tau=0.20", args.weighted, base_accuracy),
        summary_row("Post-SFT unmasked", args.unmasked, base_accuracy),
    ]
    value = {
        "base_reused_not_rerun": False,
        "eval_set": "HuggingFaceH4/MATH-500@6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "scorer": SCORER_VERSION,
        "rows": rows,
    }
    atomic_json(args.output_json, value)
    lines = [
        "# Qwen3-8B OpenR1-Math 40K full-SFT results",
        "",
        "| Model | Correct / 500 | Accuracy | Delta vs base | Bootstrap 95% CI | Cap hits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        low, high = row["bootstrap_95_ci"]
        lines.append(
            f"| {row['label']} | {row['correct']} / 500 | {100 * row['accuracy']:.1f}% | "
            f"{100 * row['delta_vs_base']:+.1f} pp | [{100 * low:.1f}%, {100 * high:.1f}%] | {row['cap_hits']} |"
        )
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(Path(args.output_md).read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
