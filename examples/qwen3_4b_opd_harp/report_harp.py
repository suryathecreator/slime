#!/usr/bin/env python3
"""Reproduce authoritative HARP V2 metrics and write compact comparison reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harp_answer_v2 import metrics_from_predictions


REPRODUCED_KEYS = (
    "num_eval_problems",
    "correct_count",
    "accuracy",
    "cap_hit_count",
    "cap_hit_rate",
    "cap_hit_accuracy",
    "non_cap_accuracy",
    "parse_failures",
    "average_generated_length",
    "extraction_method_counts",
    "verifier_method_counts",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_entry(specification: str) -> dict[str, Any]:
    label, directory_text = specification.split("=", 1)
    directory = Path(directory_text)
    predictions = read_jsonl(directory / "predictions.jsonl")
    stored = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    reproduced = metrics_from_predictions(predictions)
    disagreements = {key: {"stored": stored.get(key), "reproduced": reproduced.get(key)} for key in REPRODUCED_KEYS if stored.get(key) != reproduced.get(key)}
    if disagreements:
        raise ValueError(f"Metrics cannot be reproduced for {label}: {disagreements}")
    return {"label": label, "directory": str(directory), "metrics": stored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", required=True, help="LABEL=EVAL_DIRECTORY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    entries = [load_entry(specification) for specification in args.entry]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"title": args.title, "entries": entries}
    (output / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# {args.title}", "", "| Model/stage | Engine | Accuracy | Correct | Cap-hit rate | Cap-hit accuracy | Non-cap accuracy | Parse failures | Avg tokens |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in entries:
        metrics = entry["metrics"]
        engine_config = metrics.get("engine_config", {})
        resolved_graph_mode = engine_config.get("cudagraph_mode", "unknown")
        engine_mode = "eager" if engine_config.get("enforce_eager") else f"CUDA graph ({resolved_graph_mode})"
        engine = f"{engine_mode}, max-seqs={engine_config.get('max_num_seqs', 'n/a')}"
        lines.append(
            "| {label} | {engine} | {accuracy:.4f} | {correct}/{total} | {cap_rate:.4f} | {cap_accuracy:.4f} | {non_cap:.4f} | {parse_failures} | {avg:.1f} |".format(
                label=entry["label"],
                engine=engine,
                accuracy=metrics["accuracy"],
                correct=metrics["correct_count"],
                total=metrics["num_eval_problems"],
                cap_rate=metrics["cap_hit_rate"],
                cap_accuracy=metrics["cap_hit_accuracy"],
                non_cap=metrics["non_cap_accuracy"],
                parse_failures=metrics["parse_failures"],
                avg=metrics["average_generated_length"],
            )
        )
    lines.extend(["", "All rows use HARP seed 42, `harp_answer_v2`, and the canonical boxed-answer Qwen3 prompt.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
