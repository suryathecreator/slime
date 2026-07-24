#!/usr/bin/env python3
"""Rescore immutable MATH-500 generations with the current versioned scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen3_8b_32b_full_repro.math500_scorer import SCORER_VERSION, score_response


STAGES = ("qwen3_8b_base", "qwen3_32b_teacher", "sft_200000", "opd_100pct")
EXPECTED = {
    "qwen3_8b_base": {"correct": 373, "up": 138, "down": 0},
    "qwen3_32b_teacher": {"correct": 482, "up": 6, "down": 0},
    "sft_200000": {"correct": 434, "up": 131, "down": 1},
    "opd_100pct": {"correct": 440, "up": 11, "down": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-md", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temp = Path(handle.name)
    temp.replace(path)


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(correct: list[bool], seed: int, samples: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    n = len(correct)
    estimates = [sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    ordered = sorted(estimates)
    return [ordered[int(0.025 * (samples - 1))], ordered[int(0.975 * (samples - 1))]]


def load_source(stage_dir: Path, expected: int) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any], str]:
    summary_path = stage_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing source summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    paths = sorted((stage_dir / "problems").glob("problem_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    indices = [int(row["eval_index"]) for row in rows]
    if len(rows) != expected or indices != list(range(expected)):
        raise RuntimeError(f"Expected indices 0..{expected - 1} in {stage_dir}, found {len(rows)}")
    if summary.get("stage") != stage_dir.name or int(summary.get("total", -1)) != expected:
        raise RuntimeError(f"Source summary identity mismatch: {summary_path}")
    if {row["scorer_trace"].get("scorer_version") for row in rows} != {"math500_event_scorer_v1"}:
        raise RuntimeError(f"Source is not uniformly V1-scored: {stage_dir}")
    manifest = {int(item["eval_index"]): item for item in summary.get("raw_artifacts", [])}
    if set(manifest) != set(range(expected)):
        raise RuntimeError(f"Incomplete source raw-artifact manifest: {summary_path}")
    for row, path in zip(rows, paths, strict=True):
        expected_hash = manifest[int(row["eval_index"])].get("sha256")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Source artifact hash mismatch: {path}")
        generated_hash = hashlib.sha256(str(row["response"]).encode()).hexdigest()
        if generated_hash != row["scorer_trace"].get("generated_text_sha256"):
            raise RuntimeError(f"Source generation hash mismatch: {path}")
    return rows, paths, summary, sha256_file(summary_path)


def classify_change(source: dict[str, Any], rescored: dict[str, Any]) -> list[str]:
    reasons = []
    trace = rescored["scorer_trace"]
    source_trace = source["scorer_trace"]
    if source_trace.get("official_extracted_candidate") in {r"\]", r"\[", "$$", "$"}:
        reasons.append("structural_delimiter_rejected")
    if trace.get("extraction_rule") == "boxed_group":
        reasons.append("adjacent_boxes_grouped")
    if trace.get("partial_cap_suppression_count"):
        reasons.append("partial_cap_repetition_suppressed")
    if source["cap_hit"] and trace.get("official_extracted_candidate"):
        reasons.append("cap_strong_answer_retained")
    if source_trace.get("extraction_rule") == "standalone_final_math_line" and source["cap_hit"]:
        reasons.append("weak_cap_fallback_rejected")
    if source_trace.get("official_extracted_candidate") != trace.get("official_extracted_candidate") and not reasons:
        reasons.append("answer_event_selection_corrected")
    return reasons or ["typed_verification_corrected"]


def rescore_stage(
    source_root: Path,
    output_root: Path,
    stage: str,
    expected: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_dir = source_root / stage
    rows, paths, source_summary, source_summary_sha256 = load_source(source_dir, expected)
    stage_out = output_root / stage
    problem_out = stage_out / "problems"
    problem_out.mkdir(parents=True, exist_ok=True)
    compact_rows: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    for source, source_path in zip(rows, paths, strict=True):
        trace = score_response(
            source["response"],
            source["gold_answer"],
            finish_reason=source.get("finish_reason"),
            stop_token_id=source.get("stop_token_id"),
            cap_hit=bool(source.get("cap_hit")),
            problem_id=source["eval_index"],
        )
        result = {
            "artifact_schema_version": "math500_rescore_v2",
            "eval_index": int(source["eval_index"]),
            "stage": stage,
            "gold_answer": source["gold_answer"],
            "cap_hit": bool(source["cap_hit"]),
            "finish_reason": source.get("finish_reason"),
            "stop_token_id": source.get("stop_token_id"),
            "generated_tokens": int(source["generated_tokens"]),
            "generated_text_sha256": trace["generated_text_sha256"],
            "source_artifact": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "scorer_version": source["scorer_trace"]["scorer_version"],
            },
            "source_is_correct": bool(source["scorer_trace"]["is_correct"]),
            "source_candidate": source["scorer_trace"].get("official_extracted_candidate"),
            "scorer_trace": trace,
        }
        out_path = problem_out / f"problem_{int(source['eval_index']):04d}.json"
        atomic_text(out_path, json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        compact = {
            "eval_index": result["eval_index"],
            "source_is_correct": result["source_is_correct"],
            "is_correct": bool(trace["is_correct"]),
            "source_candidate": result["source_candidate"],
            "candidate": trace["official_extracted_candidate"],
            "candidate_sha256": trace["official_candidate_sha256"],
            "generated_text_sha256": trace["generated_text_sha256"],
            "cap_hit": result["cap_hit"],
            "extraction_rule": trace["extraction_rule"],
            "think_closed": bool(trace["think_closure"]),
            "parser_verifier_exception": trace["parser_verifier_exception"],
            "partial_cap_suppression_count": int(trace["partial_cap_suppression_count"]),
        }
        compact_rows.append(compact)
        if compact["source_is_correct"] != compact["is_correct"]:
            flips.append(
                {
                    **compact,
                    "direction": "incorrect_to_correct" if compact["is_correct"] else "correct_to_incorrect",
                    "reasons": classify_change(source, result),
                    "source_artifact_sha256": result["source_artifact"]["sha256"],
                    "selection_history": trace["selection_history"],
                }
            )
    correct = [bool(row["is_correct"]) for row in compact_rows]
    lengths = [int(row["generated_tokens"]) for row in rows]
    up = sum(not row["source_is_correct"] and row["is_correct"] for row in compact_rows)
    down = sum(row["source_is_correct"] and not row["is_correct"] for row in compact_rows)
    summary: dict[str, Any] = {
        "stage": stage,
        "scorer_version": SCORER_VERSION,
        "source_scorer_version": "math500_event_scorer_v1",
        "source_summary_path": str(source_dir / "summary.json"),
        "source_summary_sha256": source_summary_sha256,
        "source_correct": int(source_summary["correct"]),
        "correct": sum(correct),
        "total": len(correct),
        "pass_at_1": sum(correct) / len(correct),
        "bootstrap_95_ci": bootstrap_ci(correct, bootstrap_seed),
        "incorrect_to_correct": up,
        "correct_to_incorrect": down,
        "cap_hits": sum(bool(row["cap_hit"]) for row in rows),
        "think_closed": sum(bool(row["think_closed"]) for row in compact_rows),
        "eligible_final_answers": sum(row["candidate"] is not None for row in compact_rows),
        "parser_verifier_failures": sum(bool(row["parser_verifier_exception"]) for row in compact_rows),
        "partial_cap_suppressions": sum(int(row["partial_cap_suppression_count"]) for row in compact_rows),
        "length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in rows)),
        "per_problem": compact_rows,
    }
    atomic_text(stage_out / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(stage_out / "per_problem.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in compact_rows))
    atomic_text(
        stage_out / "summary.md",
        f"# {stage} MATH-500 scorer V2\n\n"
        f"- Before: {summary['source_correct']}/{summary['total']}\n"
        f"- After: {summary['correct']}/{summary['total']} ({summary['pass_at_1']:.2%})\n"
        f"- Incorrect to correct: {up}\n"
        f"- Correct to incorrect: {down}\n"
        f"- Cap hits (unchanged): {summary['cap_hits']}\n",
    )
    return summary, flips


def render_audit(audit: dict[str, Any]) -> str:
    lines = [
        "# MATH-500 scorer V2 saved-generation audit",
        "",
        "All 2,000 immutable generations were rescored. No model inference or continuation was performed.",
        "",
        "| Stage | V1 | V2 | Incorrect -> correct | Correct -> incorrect |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        item = audit["stages"][stage]
        lines.append(
            f"| {stage} | {item['source_correct']}/500 | {item['correct']}/500 ({item['pass_at_1']:.2%}) | "
            f"{item['incorrect_to_correct']} | {item['correct_to_incorrect']} |"
        )
    lines.extend(
        [
            "",
            "## Audit findings",
            "",
            "- Structural display delimiters no longer replace valid boxed answers.",
            "- Adjacent boxes are one exact-cardinality multipart answer.",
            "- Cap-hit doubt/checking language does not retract a complete answer; only an explicit unreplaced rejection does.",
            "- A terminal strict prefix of a repeated scalar or multipart answer is treated as cap truncation, not a replacement.",
            "- Natural-stop prose cannot supersede a stronger boxed or explicit answer event.",
            "- Weak standalone lines are never accepted for cap-hit rows.",
            "",
            "## Targeted audit rows",
            "",
            "| Stage / row | V1 | V2 | V2 candidate | Finding |",
            "|---|---:|---:|---|---|",
        ]
    )
    for item in audit["spot_checks"]:
        candidate = str(item["v2_candidate"] or "none").replace("|", "\\|")
        lines.append(
            f"| {item['stage']} / {item['eval_index']} | "
            f"{'correct' if item['v1_correct'] else 'incorrect'} | "
            f"{'correct' if item['v2_correct'] else 'incorrect'} | `{candidate}` | "
            f"{item['finding']} |"
        )
    lines.extend(
        [
            "",
            f"Review count: {audit['review_count']}. Aggregate gate: {audit['aggregate_gate_status']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summaries: dict[str, Any] = {}
    all_flips: dict[str, list[dict[str, Any]]] = {}
    mismatches: list[str] = []
    for stage in STAGES:
        summary, flips = rescore_stage(
            args.source_root, args.output_root, stage, args.expected, args.bootstrap_seed
        )
        summaries[stage] = summary
        all_flips[stage] = flips
        expected = EXPECTED[stage]
        observed = {
            "correct": summary["correct"],
            "up": summary["incorrect_to_correct"],
            "down": summary["correct_to_incorrect"],
        }
        if observed != expected:
            mismatches.append(f"{stage}: observed={observed} expected={expected}")
    if mismatches:
        raise RuntimeError("Audit gate mismatch after all stages were written: " + "; ".join(mismatches))
    spot_check_specs = (
        (
            "sft_200000",
            36,
            "A hypothetical 1, 2, 3 formatting example is not an answer event; the last eligible final answer remains 3, 5, 7.",
        ),
        (
            "sft_200000",
            355,
            "The cap tail contains only incidental calculation text around 100 and no strong answer event.",
        ),
        (
            "opd_100pct",
            462,
            "The terminal 5 is a strict-prefix truncation of a repeated complete answer 59, so 59 is retained.",
        ),
        (
            "qwen3_32b_teacher",
            26,
            "Later checking and a distant subproblem negation do not retract the explicit answer 144.",
        ),
    )
    spot_checks = []
    for stage, eval_index, finding in spot_check_specs:
        result = json.loads(
            (args.output_root / stage / "problems" / f"problem_{eval_index:04d}.json").read_text()
        )
        spot_checks.append(
            {
                "stage": stage,
                "eval_index": eval_index,
                "source_artifact_sha256": result["source_artifact"]["sha256"],
                "generated_text_sha256": result["generated_text_sha256"],
                "v1_correct": result["source_is_correct"],
                "v1_candidate": result["source_candidate"],
                "v2_correct": bool(result["scorer_trace"]["is_correct"]),
                "v2_candidate": result["scorer_trace"]["official_extracted_candidate"],
                "selection_history": result["scorer_trace"]["selection_history"],
                "finding": finding,
            }
        )
    audit = {
        "schema_version": 1,
        "scorer_version": SCORER_VERSION,
        "source_scorer_version": "math500_event_scorer_v1",
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "expected_rows": len(STAGES) * args.expected,
        "rescored_rows": len(STAGES) * args.expected,
        "review_count": 0,
        "aggregate_gate_status": "passed",
        "methodology": (
            "The scorer policy was derived from a quick manual scan of all 2,000 saved generations. "
            "This job revalidates immutable source hashes and applies that policy without inference."
        ),
        "expected": EXPECTED,
        "stages": summaries,
        "decision_changes": all_flips,
        "spot_checks": spot_checks,
    }
    atomic_text(args.audit_json, json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(args.audit_md, render_audit(audit))
    atomic_text(args.output_root / "_SUCCESS", json.dumps({"scorer_version": SCORER_VERSION, "audit_sha256": sha256_file(args.audit_json)}, sort_keys=True) + "\n")
    print(json.dumps({stage: {"correct": summaries[stage]["correct"], "total": summaries[stage]["total"]} for stage in STAGES}, indent=2))


if __name__ == "__main__":
    main()
