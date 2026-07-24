#!/usr/bin/env python3
"""Rescore immutable MATH-500 generations with strict boxed-answer scorer V3."""

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
V1_VERSION = "math500_event_scorer_v1"
V2_VERSION = "math500_event_scorer_v2"
EXPECTED_TAG_COUNTS = {
    "qwen3_8b_base": {
        "none": 500,
        "complete": 0,
        "open_without_close": 0,
        "close_without_open": 0,
        "malformed_or_repeated": 0,
    },
    "qwen3_32b_teacher": {
        "none": 0,
        "complete": 487,
        "open_without_close": 13,
        "close_without_open": 0,
        "malformed_or_repeated": 0,
    },
    "sft_200000": {
        "none": 0,
        "complete": 264,
        "open_without_close": 181,
        "close_without_open": 0,
        "malformed_or_repeated": 55,
    },
    "opd_100pct": {
        "none": 135,
        "complete": 0,
        "open_without_close": 0,
        "close_without_open": 359,
        "malformed_or_repeated": 6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-md", type=Path, required=True)
    parser.add_argument("--mode", choices=("draft", "final"), required=True)
    parser.add_argument("--expected-metrics", type=Path)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--expected", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    args = parser.parse_args()
    if args.mode == "final" and (not args.expected_metrics or not args.review_manifest):
        parser.error("--expected-metrics and --review-manifest are required in final mode")
    if args.mode == "draft" and (args.expected_metrics or args.review_manifest):
        parser.error("--expected-metrics and --review-manifest are only valid in final mode")
    return args


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def load_v1_source(
    stage_dir: Path,
    expected: int,
) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any], str]:
    summary_path = stage_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing V1 source summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    paths = sorted((stage_dir / "problems").glob("problem_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    indices = [int(row["eval_index"]) for row in rows]
    if len(rows) != expected or indices != list(range(expected)):
        raise RuntimeError(f"Expected V1 indices 0..{expected - 1} in {stage_dir}, found {len(rows)}")
    if summary.get("stage") != stage_dir.name or int(summary.get("total", -1)) != expected:
        raise RuntimeError(f"V1 source summary identity mismatch: {summary_path}")
    if {row["scorer_trace"].get("scorer_version") for row in rows} != {V1_VERSION}:
        raise RuntimeError(f"Source is not uniformly V1-scored: {stage_dir}")
    manifest = {int(item["eval_index"]): item for item in summary.get("raw_artifacts", [])}
    if set(manifest) != set(range(expected)):
        raise RuntimeError(f"Incomplete V1 raw-artifact manifest: {summary_path}")
    for row, path in zip(rows, paths, strict=True):
        expected_hash = manifest[int(row["eval_index"])].get("sha256")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"V1 source artifact hash mismatch: {path}")
        generated_hash = sha256_bytes(str(row["response"]).encode())
        if generated_hash != row["scorer_trace"].get("generated_text_sha256"):
            raise RuntimeError(f"V1 source generation hash mismatch: {path}")
    return rows, paths, summary, sha256_file(summary_path)


def load_v2_stage(
    stage_dir: Path,
    expected: int,
    source_rows: list[dict[str, Any]],
    source_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    summary_path = stage_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing V2 summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    paths = sorted((stage_dir / "problems").glob("problem_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    if len(rows) != expected or [int(row["eval_index"]) for row in rows] != list(range(expected)):
        raise RuntimeError(f"Expected V2 indices 0..{expected - 1} in {stage_dir}, found {len(rows)}")
    if summary.get("scorer_version") != V2_VERSION or int(summary.get("total", -1)) != expected:
        raise RuntimeError(f"V2 summary identity mismatch: {summary_path}")
    for source, source_path, row in zip(source_rows, source_paths, rows, strict=True):
        if row["scorer_trace"].get("scorer_version") != V2_VERSION:
            raise RuntimeError(f"V2 scorer mismatch: {stage_dir} row {row.get('eval_index')}")
        generated_hash = sha256_bytes(str(source["response"]).encode())
        if row.get("generated_text_sha256") != generated_hash:
            raise RuntimeError(f"V2 generation hash mismatch: {stage_dir} row {row.get('eval_index')}")
        source_artifact = row.get("source_artifact", {})
        if source_artifact.get("sha256") != sha256_file(source_path):
            raise RuntimeError(f"V2 source provenance mismatch: {stage_dir} row {row.get('eval_index')}")
    return rows, summary, sha256_file(summary_path)


def decision_digest(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "eval_index": row["eval_index"],
            "is_correct": row["is_correct"],
            "is_scorable": row["is_scorable"],
            "official_decision_reason": row["official_decision_reason"],
            "candidate_sha256": row["candidate_sha256"],
            "cap_hit": row["cap_hit"],
            "cap_counterfactual_correct": row["cap_counterfactual_correct"],
            "think_tag_state": row["think_tag_state"],
        }
        for row in rows
    ]
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def classify_v2_change(v2_row: dict[str, Any], trace: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if trace["cap_status"]:
        reasons.append("cap_hit_forced_wrong")
    reasons.append("whole_response_regardless_of_think_tags")
    v2_candidate = v2_row["scorer_trace"].get("official_extracted_candidate")
    cap_diagnostic = trace.get("cap_counterfactual_diagnostic") or {}
    v3_candidate = (
        cap_diagnostic.get("candidate")
        if trace["cap_status"]
        else trace.get("official_extracted_candidate")
    )
    if v2_candidate != v3_candidate:
        reasons.append("strict_last_box_selection")
    if v2_candidate and not v3_candidate and not trace["cap_status"]:
        reasons.append("unboxed_fallback_removed")
    if not reasons:
        reasons.append("typed_verification_result")
    return reasons


def validate_trace(trace: dict[str, Any], source: dict[str, Any]) -> None:
    if trace["scorer_version"] != SCORER_VERSION:
        raise RuntimeError("Unexpected V3 scorer version")
    if trace["generated_text_sha256"] != sha256_bytes(str(source["response"]).encode()):
        raise RuntimeError(f"V3 generation hash mismatch for row {source['eval_index']}")
    if bool(source["cap_hit"]):
        if trace["is_correct"] or trace["is_scorable"] or trace["official_extracted_candidate"] is not None:
            raise RuntimeError(f"Cap row received an official score: {source['eval_index']}")
        if trace["official_decision_reason"] != "cap_hit":
            raise RuntimeError(f"Cap row has wrong decision reason: {source['eval_index']}")
        if trace["cap_counterfactual_diagnostic"] is None:
            raise RuntimeError(f"Cap row lacks counterfactual diagnostic: {source['eval_index']}")
    elif trace["cap_counterfactual_diagnostic"] is not None:
        raise RuntimeError(f"Non-cap row has cap diagnostic: {source['eval_index']}")
    if trace["extraction_region_span"][0] != 0:
        raise RuntimeError(f"V3 did not use the whole response: {source['eval_index']}")
    if trace["extraction_region"] != "whole_response_all_think_states":
        raise RuntimeError(f"Unexpected V3 extraction region: {source['eval_index']}")
    selected = trace.get("selected_box_group")
    groups = trace.get("boxed_groups", [])
    if selected != (groups[-1] if groups else None):
        raise RuntimeError(f"Selected group is not chronologically last: {source['eval_index']}")
    if selected:
        region_start, region_end = trace["extraction_region_span"]
        start, end = selected["span"]
        if not (region_start <= start < end <= region_end):
            raise RuntimeError(f"Selected box lies outside eligible region: {source['eval_index']}")


def rescore_stage(
    source_root: Path,
    v2_root: Path,
    output_root: Path,
    stage: str,
    expected: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows, source_paths, v1_summary, v1_summary_sha256 = load_v1_source(
        source_root / stage, expected
    )
    v2_rows, v2_summary, v2_summary_sha256 = load_v2_stage(
        v2_root / stage, expected, source_rows, source_paths
    )
    stage_out = output_root / stage
    problem_out = stage_out / "problems"
    problem_out.mkdir(parents=True, exist_ok=True)
    compact_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []

    for source, source_path, v2_row in zip(source_rows, source_paths, v2_rows, strict=True):
        trace = score_response(
            source["response"],
            source["gold_answer"],
            finish_reason=source.get("finish_reason"),
            stop_token_id=source.get("stop_token_id"),
            cap_hit=bool(source.get("cap_hit")),
            problem_id=source["eval_index"],
        )
        validate_trace(trace, source)
        result = {
            "artifact_schema_version": "math500_rescore_v3",
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
                "scorer_version": V1_VERSION,
            },
            "v1_is_correct": bool(source["scorer_trace"]["is_correct"]),
            "v1_candidate": source["scorer_trace"].get("official_extracted_candidate"),
            "v2_is_correct": bool(v2_row["scorer_trace"]["is_correct"]),
            "v2_candidate": v2_row["scorer_trace"].get("official_extracted_candidate"),
            "v2_artifact": {
                "path": str(v2_root / stage / "problems" / f"problem_{int(source['eval_index']):04d}.json"),
                "sha256": sha256_file(
                    v2_root / stage / "problems" / f"problem_{int(source['eval_index']):04d}.json"
                ),
                "scorer_version": V2_VERSION,
            },
            "scorer_trace": trace,
        }
        out_path = problem_out / f"problem_{int(source['eval_index']):04d}.json"
        atomic_text(out_path, json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        cap_diagnostic = trace.get("cap_counterfactual_diagnostic") or {}
        compact = {
            "eval_index": result["eval_index"],
            "v1_is_correct": result["v1_is_correct"],
            "v2_is_correct": result["v2_is_correct"],
            "is_correct": bool(trace["is_correct"]),
            "is_scorable": bool(trace["is_scorable"]),
            "official_decision_reason": trace["official_decision_reason"],
            "v1_candidate": result["v1_candidate"],
            "v2_candidate": result["v2_candidate"],
            "candidate": trace["official_extracted_candidate"],
            "candidate_sha256": trace["official_candidate_sha256"],
            "generated_text_sha256": trace["generated_text_sha256"],
            "cap_hit": result["cap_hit"],
            "cap_counterfactual_scorable": bool(cap_diagnostic.get("is_scorable")),
            "cap_counterfactual_correct": bool(cap_diagnostic.get("is_correct")),
            "cap_counterfactual_candidate": cap_diagnostic.get("candidate"),
            "extraction_rule": trace["extraction_rule"],
            "extraction_region": trace["extraction_region"],
            "think_tag_state": trace["think_tag_state"],
            "box_group_count": len(trace["boxed_groups"]),
            "parser_verifier_exception": trace["parser_verifier_exception"],
        }
        compact_rows.append(compact)
        response = str(source["response"])
        region_start, _region_end = trace["extraction_region_span"]
        audit_row = {
            **compact,
            "stage": stage,
            "gold_answer": source["gold_answer"],
            "selected_box_group": trace["selected_box_group"],
            "malformed_boxes": trace["invalidation_flags"]["malformed_boxes"],
            "source_artifact_sha256": result["source_artifact"]["sha256"],
            "response_prefix": response[:240],
            "response_tail": response[-600:],
            "eligible_region_tail": response[region_start:][-600:],
        }
        audit_rows.append(audit_row)
        if compact["v2_is_correct"] != compact["is_correct"]:
            changes.append(
                {
                    **audit_row,
                    "direction": (
                        "incorrect_to_correct" if compact["is_correct"] else "correct_to_incorrect"
                    ),
                    "reasons": classify_v2_change(v2_row, trace),
                }
            )

    tag_counts = dict(Counter(row["think_tag_state"] for row in compact_rows))
    normalized_tag_counts = {
        state: int(tag_counts.get(state, 0)) for state in EXPECTED_TAG_COUNTS[stage]
    }
    if normalized_tag_counts != EXPECTED_TAG_COUNTS[stage]:
        raise RuntimeError(
            f"Think-tag anchor mismatch for {stage}: "
            f"observed={normalized_tag_counts} expected={EXPECTED_TAG_COUNTS[stage]}"
        )
    correct = [bool(row["is_correct"]) for row in compact_rows]
    lengths = [int(row["generated_tokens"]) for row in source_rows]
    reasons = Counter(row["official_decision_reason"] for row in compact_rows)
    v2_up = sum(not row["v2_is_correct"] and row["is_correct"] for row in compact_rows)
    v2_down = sum(row["v2_is_correct"] and not row["is_correct"] for row in compact_rows)
    unscorable_breakdown = {
        reason: count for reason, count in sorted(reasons.items()) if reason.startswith("unscorable_")
    }
    summary: dict[str, Any] = {
        "stage": stage,
        "scorer_version": SCORER_VERSION,
        "source_scorer_version": V1_VERSION,
        "prior_scorer_version": V2_VERSION,
        "source_summary_path": str(source_root / stage / "summary.json"),
        "source_summary_sha256": v1_summary_sha256,
        "v2_summary_path": str(v2_root / stage / "summary.json"),
        "v2_summary_sha256": v2_summary_sha256,
        "v1_correct": int(v1_summary["correct"]),
        "v2_correct": int(v2_summary["correct"]),
        "correct": sum(correct),
        "total": len(correct),
        "pass_at_1": sum(correct) / len(correct),
        "bootstrap_95_ci": bootstrap_ci(correct, bootstrap_seed),
        "v2_incorrect_to_correct": v2_up,
        "v2_correct_to_incorrect": v2_down,
        "cap_hits_marked_wrong": sum(row["cap_hit"] for row in compact_rows),
        "cap_hit_counterfactual_scorable": sum(
            row["cap_hit"] and row["cap_counterfactual_scorable"] for row in compact_rows
        ),
        "cap_hit_counterfactual_correct": sum(
            row["cap_hit"] and row["cap_counterfactual_correct"] for row in compact_rows
        ),
        "non_cap_unscorable": sum(
            not row["cap_hit"] and not row["is_scorable"] for row in compact_rows
        ),
        "non_cap_unscorable_breakdown": unscorable_breakdown,
        "scorable_non_cap": sum(
            not row["cap_hit"] and row["is_scorable"] for row in compact_rows
        ),
        "eligible_final_answers": sum(
            not row["cap_hit"] and row["candidate"] is not None for row in compact_rows
        ),
        "think_tag_states": normalized_tag_counts,
        "official_decision_reasons": dict(sorted(reasons.items())),
        "parser_verifier_failures": sum(
            not row["cap_hit"] and bool(row["parser_verifier_exception"]) for row in compact_rows
        ),
        "multi_box_group_rows": sum(row["box_group_count"] > 1 for row in compact_rows),
        "decision_digest": decision_digest(compact_rows),
        "length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in source_rows)),
        "per_problem": compact_rows,
    }
    if sum(summary["official_decision_reasons"].values()) != expected:
        raise RuntimeError(f"Decision reasons do not partition {stage}")
    if summary["correct"] + sum(
        count for reason, count in reasons.items() if reason != "correct"
    ) != expected:
        raise RuntimeError(f"Correct/incorrect reasons do not reconcile for {stage}")

    atomic_text(stage_out / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(
        stage_out / "per_problem.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in compact_rows),
    )
    atomic_text(
        stage_out / "audit_rows.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audit_rows),
    )
    atomic_text(
        stage_out / "summary.md",
        f"# {stage} MATH-500 strict boxed scorer V3\n\n"
        f"- V1: {summary['v1_correct']}/{summary['total']}\n"
        f"- V2: {summary['v2_correct']}/{summary['total']}\n"
        f"- V3: {summary['correct']}/{summary['total']} ({summary['pass_at_1']:.2%})\n"
        f"- Cap hits marked wrong: {summary['cap_hits_marked_wrong']}\n"
        f"- Cap-hit counterfactual correct: {summary['cap_hit_counterfactual_correct']}\n"
        f"- Non-cap unscorable: {summary['non_cap_unscorable']}\n"
        f"- Think-tag states: {json.dumps(summary['think_tag_states'], sort_keys=True)}\n",
    )
    return summary, changes, audit_rows


def expected_projection(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "correct",
        "v2_incorrect_to_correct",
        "v2_correct_to_incorrect",
        "cap_hits_marked_wrong",
        "cap_hit_counterfactual_scorable",
        "cap_hit_counterfactual_correct",
        "non_cap_unscorable",
        "scorable_non_cap",
        "think_tag_states",
        "official_decision_reasons",
        "decision_digest",
    )
    return {
        "schema_version": 1,
        "scorer_version": SCORER_VERSION,
        "rows": len(STAGES) * 500,
        "stages": {
            stage: {key: summaries[stage][key] for key in keys}
            for stage in STAGES
        },
    }


def validate_expected(
    summaries: dict[str, dict[str, Any]],
    expected_path: Path,
) -> dict[str, Any]:
    expected = json.loads(expected_path.read_text())
    observed = expected_projection(summaries)
    if observed != expected:
        raise RuntimeError(
            "Final aggregate/decision gate mismatch: "
            f"observed={json.dumps(observed, sort_keys=True)} "
            f"expected={json.dumps(expected, sort_keys=True)}"
        )
    return expected


def validate_review(
    review_path: Path,
    expected_path: Path,
    audit_record_digest: str,
    expected_rows: int,
) -> dict[str, Any]:
    review = json.loads(review_path.read_text())
    checks = {
        "schema_version": review.get("schema_version") == 1,
        "scorer_version": review.get("scorer_version") == SCORER_VERSION,
        "reviewed_rows": int(review.get("reviewed_rows", -1)) == expected_rows,
        "unresolved_review_count": int(review.get("unresolved_review_count", -1)) == 0,
        "audit_record_digest": review.get("audit_record_digest") == audit_record_digest,
        "expected_metrics_sha256": review.get("expected_metrics_sha256") == sha256_file(expected_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Final review-manifest gate failed: {failed}")
    return review


def render_audit(audit: dict[str, Any]) -> str:
    lines = [
        "# MATH-500 strict boxed scorer V3 saved-generation audit",
        "",
        "All 2,000 immutable generations were rescored without inference or continuation.",
        "Cap hits are officially wrong; their strict-box result is retained only as a diagnostic.",
        "",
        "| Stage | V1 | V2 | V3 strict | Cap wrong | Cap diagnostic correct | Non-cap unscorable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        item = audit["stages"][stage]
        lines.append(
            f"| {stage} | {item['v1_correct']}/500 | {item['v2_correct']}/500 | "
            f"{item['correct']}/500 ({item['pass_at_1']:.2%}) | "
            f"{item['cap_hits_marked_wrong']} | {item['cap_hit_counterfactual_correct']} | "
            f"{item['non_cap_unscorable']} |"
        )
    lines.extend(
        [
            "",
            "## Think-tag formatting diagnostics",
            "",
            "| Stage | Complete | None | Open/no close | Close/no open | Malformed/repeated |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGES:
        states = audit["stages"][stage]["think_tag_states"]
        lines.append(
            f"| {stage} | {states['complete']} | {states['none']} | "
            f"{states['open_without_close']} | {states['close_without_open']} | "
            f"{states['malformed_or_repeated']} |"
        )
    lines.extend(
        [
            "",
            "The model is still learning consistent think-tag behavior while its reasoning is improving. "
            "These counts measure formatting, not reasoning quality; substantially more instruction-tuning data is "
            "likely needed to teach consistent formatting.",
            "",
            "Think tags never affect extraction. Every row uses the whole response, and the scorer takes the chronologically last "
            "complete boxed group and does not interpret later reasoning, doubt, retraction, examples, or prose.",
            "",
            f"Review count: {audit['review_count']}. Aggregate gate: {audit['aggregate_gate_status']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summaries: dict[str, dict[str, Any]] = {}
    all_changes: dict[str, list[dict[str, Any]]] = {}
    all_audit_rows: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(STAGES):
        summary, changes, audit_rows = rescore_stage(
            args.source_root,
            args.v2_root,
            args.output_root,
            stage,
            args.expected,
            args.bootstrap_seed + stage_index,
        )
        summaries[stage] = summary
        all_changes[stage] = changes
        all_audit_rows.extend(audit_rows)

    projection = expected_projection(summaries)
    audit_record_digest = sha256_bytes(
        json.dumps(all_audit_rows, sort_keys=True, separators=(",", ":")).encode()
    )
    review: dict[str, Any] | None = None
    if args.mode == "final":
        validate_expected(summaries, args.expected_metrics)
        review = validate_review(
            args.review_manifest,
            args.expected_metrics,
            audit_record_digest,
            len(STAGES) * args.expected,
        )
        aggregate_status = "passed"
    else:
        atomic_text(
            args.output_root / "suggested_expected_metrics.json",
            json.dumps(projection, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        aggregate_status = "draft_pending_manual_audit"

    audit = {
        "schema_version": 3,
        "scorer_version": SCORER_VERSION,
        "source_scorer_version": V1_VERSION,
        "prior_scorer_version": V2_VERSION,
        "source_root": str(args.source_root),
        "v2_root": str(args.v2_root),
        "output_root": str(args.output_root),
        "mode": args.mode,
        "expected_rows": len(STAGES) * args.expected,
        "rescored_rows": sum(item["total"] for item in summaries.values()),
        "review_count": int(review["unresolved_review_count"]) if review else None,
        "aggregate_gate_status": aggregate_status,
        "methodology": (
            "Every immutable source and generation hash was validated. The draft emits compact audit "
            "records for all 2,000 rows; publication requires a separately locked aggregate and row-decision "
            "digest produced only after the compact scan and deep review of exceptional rows."
        ),
        "think_tag_interpretation": (
            "Think-tag counts measure formatting, not reasoning quality. The model is still learning tag "
            "behavior while reasoning improves, and substantially more instruction-tuning data is likely needed for "
            "consistent formatting. Think tags never affect V3 extraction; every row uses the whole response."
        ),
        "expected_tag_counts": EXPECTED_TAG_COUNTS,
        "stages": summaries,
        "decision_changes_from_v2": all_changes,
        "audit_record_count": len(all_audit_rows),
        "audit_record_digest": audit_record_digest,
        "review_manifest": (
            {
                "path": str(args.review_manifest),
                "sha256": sha256_file(args.review_manifest),
                "draft_job_id": review["draft_job_id"],
                "draft_launch_commit": review["draft_launch_commit"],
                "reviewed_rows": review["reviewed_rows"],
                "review_scope": review["review_scope"],
                "findings": review["findings"],
                "unresolved_review_count": review["unresolved_review_count"],
            }
            if review
            else None
        ),
    }
    atomic_text(args.audit_json, json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(args.audit_md, render_audit(audit))
    marker = "_SUCCESS" if args.mode == "final" else "_DRAFT"
    atomic_text(
        args.output_root / marker,
        json.dumps(
            {
                "scorer_version": SCORER_VERSION,
                "mode": args.mode,
                "audit_sha256": sha256_file(args.audit_json),
                "audit_record_digest": audit["audit_record_digest"],
            },
            sort_keys=True,
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "audit": str(args.audit_json),
                "stages": {
                    stage: {
                        "correct": summaries[stage]["correct"],
                        "total": summaries[stage]["total"],
                        "cap_hits_marked_wrong": summaries[stage]["cap_hits_marked_wrong"],
                        "non_cap_unscorable": summaries[stage]["non_cap_unscorable"],
                    }
                    for stage in STAGES
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
