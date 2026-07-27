#!/usr/bin/env python3
"""Rescore immutable AIME generation artifacts with aime_answer_v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from aime_answer_v2 import SCORER_FIELDS, SCORER_VERSION, metrics_from_predictions, score_answer


_METRIC_SCORE_FIELDS = {
    "num_generations",
    "total_correct",
    "pass_at_1_mc",
    "sampled_pass_at_1",
    "cap_hit_count",
    "cap_hit_rate",
    "cap_hit_accuracy",
    "parse_failure_count",
    "extraction_method_counts",
    "decision_reason_counts",
    "region_policy_counts",
    "selection_action_counts",
    "replacement_count",
    "reaffirmation_count",
    "retraction_count",
    "per_problem",
    "scorer_version",
    "predictions_sha256",
    "rescore_provenance",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    source_version = row.get("scorer_version")
    if source_version != "aime_answer_v1":
        raise ValueError(f"Expected aime_answer_v1 input row, found {source_version!r}")
    metadata = {key: value for key, value in row.items() if key not in SCORER_FIELDS}
    metadata["source_scorer_version"] = source_version
    metadata.update(
        score_answer(
            str(row["generated_text"]),
            row["correct_answer"],
            str(row["problem"]),
            str(row["problem_id"]),
            bool(row["cap_hit"]),
            row.get("finish_reason"),
        )
    )
    return metadata


def per_problem_metrics(rows: list[dict[str, Any]], samples_per_problem: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    problem_ids = sorted({str(row["problem_id"]) for row in rows}, key=lambda item: int(item.rsplit("_", 1)[-1]))
    for problem_id in problem_ids:
        problem_rows = [row for row in rows if row["problem_id"] == problem_id]
        if len(problem_rows) != samples_per_problem:
            raise ValueError(f"{problem_id} has {len(problem_rows)} rows, expected {samples_per_problem}")
        problem_rows.sort(key=lambda row: int(row["sample_index"]))
        if [int(row["sample_index"]) for row in problem_rows] != list(range(samples_per_problem)):
            raise ValueError(f"{problem_id} has a noncanonical sample grid")
        if samples_per_problem == 1:
            result.append(
                {
                    "problem_id": problem_id,
                    "problem_idx": int(problem_rows[0]["problem_idx"]),
                    "seed": int(problem_rows[0]["seed"]),
                    "is_correct": bool(problem_rows[0]["is_correct"]),
                }
            )
        else:
            correct = sum(bool(row["is_correct"]) for row in problem_rows)
            result.append(
                {
                    "problem_id": problem_id,
                    "problem_idx": int(problem_rows[0]["problem_idx"]),
                    "correct_count": correct,
                    "num_samples": samples_per_problem,
                    "pass_at_1_mc": correct / samples_per_problem,
                }
            )
    return result


def diff_rows(source: list[dict[str, Any]], rescored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("is_correct", "candidate", "parsed_answer", "extraction_method", "decision_reason", "region_policy")
    differences: list[dict[str, Any]] = []
    if len(source) != len(rescored):
        raise ValueError("Source and rescored prediction counts differ")
    for before, after in zip(source, rescored):
        changes = {
            field: {"v1": before.get(field), "v2": after.get(field)}
            for field in fields
            if before.get(field) != after.get(field)
        }
        if changes:
            differences.append(
                {
                    "problem_id": before["problem_id"],
                    "problem_idx": int(before["problem_idx"]),
                    "sample_index": int(before["sample_index"]),
                    "flat_index": int(before["flat_index"]),
                    "changes": changes,
                }
            )
    return differences


def rescore(input_dir: Path, output_dir: Path) -> None:
    source_predictions = input_dir / "predictions.jsonl"
    source_metrics_path = input_dir / "metrics.json"
    source_rows = read_jsonl(source_predictions)
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    source_predictions_sha256 = sha256_file(source_predictions)
    source_metrics_sha256 = sha256_file(source_metrics_path)
    if source_metrics.get("predictions_sha256") != source_predictions_sha256:
        raise ValueError("Source metrics do not match the source predictions hash")
    if source_metrics.get("scorer_version") != "aime_answer_v1":
        raise ValueError("Source metrics were not produced by aime_answer_v1")
    if int(source_metrics.get("num_generations", -1)) != len(source_rows):
        raise ValueError("Source metrics and predictions disagree on generation count")

    rescored_rows = [score_row(row) for row in source_rows]
    output_predictions = output_dir / "predictions.jsonl"
    write_jsonl_atomic(output_predictions, rescored_rows)
    output_predictions_sha256 = sha256_file(output_predictions)

    samples_per_problem = int(source_metrics["samples_per_problem"])
    reproduced = metrics_from_predictions(rescored_rows)
    output_metrics = {key: value for key, value in source_metrics.items() if key not in _METRIC_SCORE_FIELDS}
    output_metrics.update(reproduced)
    output_metrics.update(
        {
            "scorer_version": SCORER_VERSION,
            "predictions_sha256": output_predictions_sha256,
            "per_problem": per_problem_metrics(rescored_rows, samples_per_problem),
            "rescore_provenance": {
                "source_directory": str(input_dir.resolve()),
                "source_predictions_sha256": source_predictions_sha256,
                "source_metrics_sha256": source_metrics_sha256,
                "source_scorer_version": "aime_answer_v1",
                "target_scorer_version": SCORER_VERSION,
            },
        }
    )
    if samples_per_problem == 1:
        output_metrics["sampled_pass_at_1"] = reproduced["total_correct"] / len(rescored_rows)
    write_json_atomic(output_dir / "metrics.json", output_metrics)

    differences = diff_rows(source_rows, rescored_rows)
    grade_flips = [item for item in differences if "is_correct" in item["changes"]]
    audit = {
        "source": {
            "directory": str(input_dir.resolve()),
            "predictions_sha256": source_predictions_sha256,
            "metrics_sha256": source_metrics_sha256,
            "scorer_version": "aime_answer_v1",
            "total_correct": int(source_metrics["total_correct"]),
            "score": float(
                source_metrics.get("sampled_pass_at_1", source_metrics.get("pass_at_1_mc"))
            ),
        },
        "rescored": {
            "directory": str(output_dir.resolve()),
            "predictions_sha256": output_predictions_sha256,
            "metrics_sha256": sha256_file(output_dir / "metrics.json"),
            "scorer_version": SCORER_VERSION,
            "total_correct": int(reproduced["total_correct"]),
            "score": float(reproduced["pass_at_1_mc"]),
        },
        "changed_row_count": len(differences),
        "grade_flip_count": len(grade_flips),
        "grade_flips": grade_flips,
        "row_differences": differences,
    }
    write_json_atomic(output_dir / "diff_from_aime_answer_v1.json", audit)
    print(
        f"Rescored {len(rescored_rows)} rows: "
        f"v1={audit['source']['total_correct']}/{len(rescored_rows)} "
        f"v2={audit['rescored']['total_correct']}/{len(rescored_rows)} "
        f"grade_flips={len(grade_flips)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Refusing to overwrite the source scorer artifacts")
    rescore(input_dir, output_dir)


if __name__ == "__main__":
    main()
