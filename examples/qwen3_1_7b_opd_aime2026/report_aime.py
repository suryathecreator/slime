#!/usr/bin/env python3
"""Validate AIME 2026 metrics and write compact stage comparison reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from aime_answer_v2 import SCORER_VERSION, metrics_from_predictions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_entry(specification: str) -> dict[str, Any]:
    label, directory_text = specification.split("=", 1)
    directory = Path(directory_text)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(directory / "predictions.jsonl")
    if metrics.get("scorer_version") != SCORER_VERSION:
        raise ValueError(
            f"{label} uses scorer {metrics.get('scorer_version')!r}; expected {SCORER_VERSION!r}"
        )
    if any(row.get("scorer_version") != SCORER_VERSION for row in predictions):
        raise ValueError(f"{label} predictions contain a scorer version other than {SCORER_VERSION}")
    reproduced = metrics_from_predictions(predictions)
    for key in ("num_generations", "total_correct", "pass_at_1_mc", "cap_hit_count", "cap_hit_rate", "cap_hit_accuracy", "parse_failure_count", "extraction_method_counts", "decision_reason_counts"):
        if metrics.get(key) != reproduced.get(key):
            raise ValueError(f"Cannot reproduce {key} for {label}: stored={metrics.get(key)!r}, reproduced={reproduced.get(key)!r}")
    if metrics.get("num_problems") != 30 or metrics.get("samples_per_problem") != 16 or metrics.get("num_generations") != 480:
        raise ValueError(f"{label} does not implement the required 30x16 AIME protocol")
    if metrics.get("metric_name") != "pass_at_1_mc_16":
        raise ValueError(f"{label} reports the wrong metric semantics")
    per_problem = metrics.get("per_problem") or []
    if len(per_problem) != 30 or any(int(row["num_samples"]) != 16 for row in per_problem):
        raise ValueError(f"{label} has incomplete per-problem sampling metrics")
    return {"label": label, "directory": str(directory), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", required=True, help="LABEL=EVAL_DIRECTORY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    entries = [load_entry(specification) for specification in args.entry]
    output = Path(args.output_dir)
    payload = {
        "title": args.title,
        "metric_semantics": "16-sample Monte Carlo estimate of single-sample accuracy; not pass@16 and not majority vote",
        "entries": entries,
    }
    write_atomic(output / "report.json", json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    lines = [
        f"# {args.title}",
        "",
        "| Model/stage | MC pass@1 | Correct generations | Cap-hit rate | Cap-hit accuracy | Avg tokens | Aggregate tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        metrics = entry["metrics"]
        lines.append(
            "| {label} | {pass1:.4f} | {correct}/480 | {cap_rate:.4f} | {cap_accuracy:.4f} | {avg:.1f} | {throughput:.1f} |".format(
                label=entry["label"],
                pass1=metrics["pass_at_1_mc"],
                correct=metrics["total_correct"],
                cap_rate=metrics["cap_hit_rate"],
                cap_accuracy=metrics["cap_hit_accuracy"],
                avg=metrics["average_generated_length"],
                throughput=metrics["aggregate_generated_tokens_per_second"],
            )
        )
    lines.extend(
        [
            "",
            "Each benchmark value is total correct divided by 480: 30 problems with 16 independent sampled generations each.",
            "Sampling is temperature 0.6, top_p 0.95, top_k 20, min_p 0, with deterministic per-request seeds beginning at 42.",
            "This estimates single-sample accuracy; it is neither pass@16 nor majority-vote accuracy.",
            "",
        ]
    )
    write_atomic(output / "report.md", "\n".join(lines))


if __name__ == "__main__":
    main()
