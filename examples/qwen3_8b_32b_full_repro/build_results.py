#!/usr/bin/env python3
"""Build immutable scratch result and stopping-audit summaries after final eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


STAGES = ("qwen3_8b_base", "qwen3_32b_teacher", "sft_200000", "opd_100pct")
def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage(root: Path, stage: str) -> dict[str, Any]:
    path = root / stage / "summary.json"
    if not path.is_file():
        raise SystemExit(f"Missing required final evaluation: {path}")
    value = json.loads(path.read_text())
    value["summary_path"] = str(path)
    value["summary_sha256"] = sha256_file(path)
    return value


def stopping_audit(root: Path) -> dict[str, Any]:
    rollout_audit = Path(os.environ.get("OPD_ROLLOUT_AUDIT_JSON", ""))
    if rollout_audit.is_file():
        audit = json.loads(rollout_audit.read_text())
        audit["status"] = "complete"
        audit["source"] = str(rollout_audit)
        atomic_text(root / "STOPPING_OBJECTIVE_AUDIT.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
        source_md = Path(os.environ["OPD_ROLLOUT_AUDIT_MD"])
        atomic_text(root / "STOPPING_OBJECTIVE_AUDIT.md", source_md.read_text())
        return audit
    sanity_dir = Path(os.environ["OPD_SANITY_REPORT_DIR"])
    reports = [json.loads(path.read_text()) for path in sorted(sanity_dir.glob("rollout_*.json"))]
    if not reports:
        return {"status": "missing", "reports": 0, "note": "No OPD sanity windows were found."}
    weighted_n = sum(int(report.get("n", 0)) for report in reports)
    weighted = lambda key: sum(float(report.get(key, 0)) * int(report.get("n", 0)) for report in reports) / max(1, weighted_n)
    audit = {
        "status": "complete",
        "reports": len(reports),
        "samples": weighted_n,
        "mean_response_tokens": weighted("avg_response_tokens"),
        "max_response_tokens": max(int(report.get("max_response_tokens", 0)) for report in reports),
        "cap_hit_rate": weighted("cap_hit_rate"),
        "completed_rate": weighted("completed_rate"),
        "eligible_final_answer_rate": weighted("final_answer_rate"),
        "windows_with_threshold_violations": sum(bool(report.get("violations")) for report in reports),
        "behavior_thresholds_were_fatal": False,
        "interpretation": "The 16K cutoff bounds compute and does not teach stopping; capped trajectories were retained in the faithful loss.",
    }
    atomic_text(root / "STOPPING_OBJECTIVE_AUDIT.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    atomic_text(
        root / "STOPPING_OBJECTIVE_AUDIT.md",
        "# Stopping-objective audit\n\n"
        f"- Rollout windows: {audit['reports']}\n"
        f"- Trajectories observed: {audit['samples']}\n"
        f"- Mean/max response tokens: {audit['mean_response_tokens']:.1f}/{audit['max_response_tokens']}\n"
        f"- 16K cap-hit rate: {audit['cap_hit_rate']:.2%}\n"
        f"- Completed rate: {audit['completed_rate']:.2%}\n"
        f"- Eligible final-answer rate: {audit['eligible_final_answer_rate']:.2%}\n"
        f"- Windows with diagnostic threshold violations: {audit['windows_with_threshold_violations']}\n\n"
        "The 16K cutoff bounded compute but was not interpreted as a stopping target. Capped trajectories stayed in the faithful OPD loss, and behavior thresholds were nonfatal.\n",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path(os.environ["EVAL_OUTPUT_ROOT"]))
    parser.add_argument("--historical-eval-root", type=Path)
    parser.add_argument("--scorer-audit", type=Path)
    args = parser.parse_args()
    eval_root = args.eval_root
    report_root = Path(os.environ["OUTPUT_ROOT"]) / "final_report"
    values = {stage: load_stage(eval_root, stage) for stage in STAGES}
    historical_values = (
        {stage: load_stage(args.historical_eval_root, stage) for stage in STAGES}
        if args.historical_eval_root
        else None
    )
    scorer_audit = json.loads(args.scorer_audit.read_text()) if args.scorer_audit else None
    scorer_audit_reference = None
    if scorer_audit:
        if scorer_audit.get("aggregate_gate_status") != "passed" or scorer_audit.get("review_count") != 0:
            raise SystemExit("Scorer audit did not pass fail-closed publication gates")
        for stage in STAGES:
            if int(scorer_audit["stages"][stage]["correct"]) != int(values[stage]["correct"]):
                raise SystemExit(f"Scorer audit/result mismatch for {stage}")
        stage_keys = (
            "source_correct",
            "correct",
            "total",
            "pass_at_1",
            "incorrect_to_correct",
            "correct_to_incorrect",
            "cap_hits",
            "eligible_final_answers",
            "partial_cap_suppressions",
        )
        scorer_audit_reference = {
            "schema_version": scorer_audit["schema_version"],
            "scorer_version": scorer_audit["scorer_version"],
            "source_scorer_version": scorer_audit["source_scorer_version"],
            "audit_path": str(args.scorer_audit),
            "audit_sha256": sha256_file(args.scorer_audit),
            "expected_rows": scorer_audit["expected_rows"],
            "rescored_rows": scorer_audit["rescored_rows"],
            "review_count": scorer_audit["review_count"],
            "aggregate_gate_status": scorer_audit["aggregate_gate_status"],
            "methodology": scorer_audit["methodology"],
            "stages": {
                stage: {key: scorer_audit["stages"][stage][key] for key in stage_keys}
                for stage in STAGES
            },
            "decision_change_counts": {
                stage: len(scorer_audit["decision_changes"][stage]) for stage in STAGES
            },
            "spot_checks": scorer_audit["spot_checks"],
        }
    learning_path = report_root / "opd_learning_dynamics.json"
    attempt_32k_path = report_root / "CAP_HIT_32K_ATTEMPT.json"
    if not learning_path.is_file():
        raise SystemExit(f"Missing OPD learning dynamics: {learning_path}")
    if not attempt_32k_path.is_file():
        raise SystemExit(f"Missing 32K attempt record: {attempt_32k_path}")
    learning = json.loads(learning_path.read_text())
    attempt_32k = json.loads(attempt_32k_path.read_text())
    if attempt_32k.get("status") != "abandoned_incomplete_no_score":
        raise SystemExit(f"Unexpected 32K attempt status: {attempt_32k.get('status')}")
    sft = values["sft_200000"]
    opd = values["opd_100pct"]
    sft_points = 100 * float(sft["pass_at_1"])
    opd_points = 100 * float(opd["pass_at_1"])
    criteria = {
        "sft_target": 76.0,
        "opd_target": 94.0,
        "target_delta": 18.0,
        "sft_within_3_points": abs(sft_points - 76.0) <= 3.0,
        "opd_within_3_points": abs(opd_points - 94.0) <= 3.0,
        "opd_gain_at_least_12_points": opd_points - sft_points >= 12.0,
        "observed_gain_points": opd_points - sft_points,
    }
    audit = stopping_audit(report_root)
    result = {
        "schema_version": 2 if historical_values else 1,
        "contract_sha256": sha256_file(Path(os.environ["CONTRACT_FILE"])),
        "evaluations": values,
        "historical_evaluations": historical_values,
        "scorer_audit": scorer_audit_reference,
        "cap_hit_32k_context_evaluation": attempt_32k,
        "reproduction_criteria": criteria,
        "stopping_audit": audit,
        "opd_learning_dynamics": learning,
    }
    atomic_text(report_root / "results.json", json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    lines = ["# Qwen3 8B -> 32B reproduction results", ""]
    if historical_values:
        lines.extend(
            [
                "The primary results use `math500_event_scorer_v2`; the original V1 scores are retained for provenance.",
                "",
                "| Stage | V1 before audit | V2 corrected |",
                "|---|---:|---:|",
            ]
        )
        for stage in STAGES:
            before = historical_values[stage]
            after = values[stage]
            lines.append(
                f"| {stage} | {before['correct']}/500 ({100 * before['pass_at_1']:.2f}%) | "
                f"{after['correct']}/500 ({100 * after['pass_at_1']:.2f}%) |"
            )
        lines.extend(
            [
                "",
                "All 2,000 saved generations were rescored without new inference. See `MATH500_SCORER_AUDIT.md` for row-level changes and bug categories.",
            ]
        )
    else:
        for stage in STAGES:
            value = values[stage]
            lines.append(f"- {stage}: {value['correct']}/500 ({100 * value['pass_at_1']:.2f}%)")
    lines.extend(["", "## OPD learning dynamics", ""])
    lines.extend(
        [
            "| Rollouts | Mean tokens | Cap hit | Think closed | Eligible final | `<|im_end|>` | Repetition | Reverse-KL |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in learning["milestone_windows"]:
        lines.append(
            f"| {row['rollout_start']}-{row['rollout_end']} | {row['length']['mean']:.1f} | "
            f"{row['cap_hit_rate']:.2%} | {row['think_closed_rate']:.2%} | "
            f"{row['eligible_final_answer_rate']:.2%} | "
            f"{row['terminal_condition_rates'].get('im_end', 0.0):.2%} | "
            f"{row['repetition_indicator_rate']:.2%} | {row['sampled_reverse_kl']:.4f} |"
        )
    delta = learning["first_to_last_window_change"]
    lines.extend(
        [
            "",
            "These are descriptive windows over different OpenThoughts prompts, not a controlled within-prompt estimate or a correctness measurement. From the first to final window, "
            f"think closure changed by {delta['think_closed_rate_percentage_points']:+.2f} points, "
            f"natural stopping by {delta['natural_stop_rate_percentage_points']:+.2f} points, "
            f"`<|im_end|>` termination by {delta['im_end_rate_percentage_points']:+.2f} points, "
            f"and cap hits by {delta['cap_hit_rate_percentage_points']:+.2f} points. "
            "The 16K cutoff bounded compute; it was not an explicit terminal target.",
            "",
            "## Abandoned 32K context attempt",
            "",
            f"Job `{attempt_32k['job']['job_id']}` selected all {attempt_32k['attempt']['selected_cap_hit_rows']} OPD cap-hit rows. "
            f"All {attempt_32k['attempt']['generation_attempts_reached_engine_completion']} generation attempts reached engine completion, "
            f"but exact-prefix replay failed and {attempt_32k['attempt']['published_rows']} rows were published. No 32K score exists.",
            "",
            attempt_32k["fidelity_decision"]["exact_replay_unreliable"],
            "",
            attempt_32k["fidelity_decision"]["prefix_continuation_rejected"],
            "",
            attempt_32k["fidelity_decision"]["required_for_valid_32k_ablation"],
            "",
            attempt_32k["fidelity_decision"]["future_work"],
            "",
            "Remaining jobs were cancelled: "
            + ", ".join(
                f"`{row['job_id']}` ({row['stage']})"
                for row in attempt_32k["cancelled_downstream_jobs"]
            )
            + ".",
        ]
    )
    lines.extend(
        [
            "",
            f"Observed OPD gain: {criteria['observed_gain_points']:.2f} points. Public targets were 76% SFT and 94% OPD; they were not used to choose row-level scorer decisions.",
            "",
            "Primary reproduction criteria use only the completed fixed-16K-output evaluations above. No incomplete 32K attempt is included in any metric.",
        ]
    )
    atomic_text(report_root / "RESULTS.md", "\n".join(lines) + "\n")
    print(json.dumps({"report_root": str(report_root), "criteria": criteria}, indent=2))


if __name__ == "__main__":
    main()
