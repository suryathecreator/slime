#!/usr/bin/env python3
"""Build four 2K MATH-500 tables with one shared set of controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.qwen3_8b_32b_full_repro.math500_scorer import SCORER_VERSION
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    atomic_json,
    sha256_file,
)


TABLES = {
    "random_token_masking": [
        "random_mask_25",
        "random_mask_50",
        "random_mask_70",
        "random_mask_80",
        "random_mask_90",
    ],
    "inverse_margin_weighting": [
        "inverse_tau_0p20",
        "inverse_tau_0p05",
    ],
    "regular_margin_masking": ["margin_mask"],
    "probability_ratio_masking": ["prob_ratio_mask"],
}
LABELS = {
    "base_8b": "Base 8B",
    "correct_only": "Post-SFT correct-only",
    "unmasked": "Post-SFT correct + wrong unmasked",
    "random_mask_25": "Random token masking 25%",
    "random_mask_50": "Random token masking 50%",
    "random_mask_70": "Random token masking 70%",
    "random_mask_80": "Random token masking 80%",
    "random_mask_90": "Random token masking 90%",
    "inverse_tau_0p20": "Inverse margin weighting tau=0.20",
    "inverse_tau_0p05": "Inverse margin weighting tau=0.05",
    "margin_mask": "Regular margin masking",
    "prob_ratio_mask": "Probability-ratio masking",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def summary_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if separator != "=" or name not in LABELS or not path or name in result:
            raise ValueError(f"invalid --summary value: {value}")
        result[name] = path
    if set(result) != set(LABELS):
        raise ValueError(
            f"summary set mismatch missing={sorted(set(LABELS) - set(result))} "
            f"extra={sorted(set(result) - set(LABELS))}"
        )
    return result


def row(name: str, path: str, base_accuracy: float) -> dict[str, Any]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(summary.get("total", -1)) != 500 or len(
        summary.get("per_problem", [])
    ) != 500:
        raise ValueError(f"incomplete MATH-500 summary: {path}")
    if summary.get("scorer_version") != SCORER_VERSION:
        raise ValueError(f"scorer mismatch: {path}")
    accuracy = float(summary["pass_at_1"])
    return {
        "variant": name,
        "label": LABELS[name],
        "correct": int(summary["correct"]),
        "total": 500,
        "accuracy": accuracy,
        "delta_vs_base": accuracy - base_accuracy,
        "bootstrap_95_ci": [float(value) for value in summary["bootstrap_95_ci"]],
        "cap_hits": int(summary["cap_hits"]),
        "summary_path": path,
        "summary_sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    summaries = summary_map(args.summary)
    if sha256_file(summaries["base_8b"]) != args.base_sha256:
        raise ValueError("shared base summary hash mismatch")
    base_value = json.loads(
        Path(summaries["base_8b"]).read_text(encoding="utf-8")
    )
    base_accuracy = float(base_value["pass_at_1"])
    rows = {
        name: row(name, path, base_accuracy) for name, path in summaries.items()
    }
    shared_names = ["base_8b", "correct_only", "unmasked"]
    tables = {
        table_name: [rows[name] for name in [*shared_names, *variant_names]]
        for table_name, variant_names in TABLES.items()
    }
    value = {
        "eval_set": "HuggingFaceH4/MATH-500@6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "scorer": SCORER_VERSION,
        "decoding": "greedy",
        "dynamic_response_budget": "32768 - rendered_prompt_tokens - 64",
        "shared_controls": shared_names,
        "tables": tables,
    }
    atomic_json(args.output_json, value)
    lines = [
        "# Qwen3-8B OpenR1-Math 2K Slime full-SFT results",
        "",
        "Base, correct-only, and correct + wrong unmasked are single evaluations reused in every table.",
    ]
    for table_name, table_rows in tables.items():
        lines.extend(
            [
                "",
                f"## {table_name.replace('_', ' ').title()}",
                "",
                "| Model | Correct / 500 | Accuracy | Delta vs base | 95% CI | Cap hits |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in table_rows:
            low, high = item["bootstrap_95_ci"]
            lines.append(
                f"| {item['label']} | {item['correct']} / 500 | "
                f"{100 * item['accuracy']:.1f}% | "
                f"{100 * item['delta_vs_base']:+.1f} pp | "
                f"[{100 * low:.1f}%, {100 * high:.1f}%] | "
                f"{item['cap_hits']} |"
            )
    target = Path(args.output_md)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
