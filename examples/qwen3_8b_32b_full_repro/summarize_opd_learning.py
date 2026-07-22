#!/usr/bin/env python3
"""Summarize how OPD rollout behavior changed across training checkpoints."""

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


MILESTONE_WINDOWS = ((0, 22), (23, 45), (46, 68), (69, 91))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-jsonl",
        type=Path,
        default=Path(os.environ["OPD_ROLLOUT_AUDIT_JSONL"]),
    )
    report_root = Path(os.environ["OUTPUT_ROOT"]) / "final_report"
    parser.add_argument(
        "--out-json",
        type=Path,
        default=report_root / "opd_learning_dynamics.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=report_root / "OPD_LEARNING_DYNAMICS.md",
    )
    parser.add_argument("--expected-rollouts", type=int, default=92)
    parser.add_argument("--expected-trajectories", type=int, default=5888)
    parser.add_argument("--samples-per-rollout", type=int, default=64)
    return parser.parse_args()


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


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def summarize_rows(rows: list[dict[str, Any]], rollout_start: int, rollout_end: int) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No rows for rollout window {rollout_start}-{rollout_end}")
    lengths = [int(row["response_tokens"]) for row in rows]
    terminal_counts = Counter(str(row.get("terminal_condition")) for row in rows)
    total = len(rows)
    count = lambda key: sum(bool(row.get(key)) for row in rows)
    eligible = sum(row.get("eligible_final_answer") is not None for row in rows)
    terminal_rates = {key: value / total for key, value in sorted(terminal_counts.items())}
    return {
        "rollout_start": rollout_start,
        "rollout_end": rollout_end,
        "trajectories": total,
        "unique_source_rows": len({row.get("source_row_id") for row in rows}),
        "length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "cap_hits": count("cap_hit"),
        "cap_hit_rate": count("cap_hit") / total,
        "natural_stop_rate": 1.0 - count("cap_hit") / total,
        "think_closed": count("think_closed"),
        "think_closed_rate": count("think_closed") / total,
        "eligible_final_answers": eligible,
        "eligible_final_answer_rate": eligible / total,
        "terminal_conditions": dict(sorted(terminal_counts.items())),
        "terminal_condition_rates": terminal_rates,
        "repetition_indicators": count("repetition_indicator"),
        "repetition_indicator_rate": count("repetition_indicator") / total,
        "doubt_rate": count("doubt") / total,
        "retraction_rate": count("retraction") / total,
        "answer_events_per_trajectory": sum(int(row.get("answer_events") or 0) for row in rows) / total,
        "answer_replacements_per_trajectory": sum(
            int(row.get("answer_replacements") or 0) for row in rows
        )
        / total,
        "sampled_reverse_kl": mean_numeric(rows, "sampled_reverse_kl"),
        "mean_teacher_terminal_logprob": mean_numeric(rows, "teacher_terminal_logprob"),
        "mean_student_terminal_logprob": mean_numeric(rows, "student_terminal_logprob"),
    }


def percentage_point_change(first: dict[str, Any], last: dict[str, Any], key: str) -> float:
    return 100.0 * (float(last[key]) - float(first[key]))


def build_summary(
    rows: list[dict[str, Any]],
    *,
    source_path: Path,
    expected_rollouts: int,
    expected_trajectories: int,
    samples_per_rollout: int,
) -> dict[str, Any]:
    rollout_ids = sorted({int(row["rollout_id"]) for row in rows})
    if rollout_ids != list(range(expected_rollouts)):
        raise ValueError(f"Rollout IDs are incomplete: {rollout_ids}")
    if len(rows) != expected_trajectories:
        raise ValueError(f"Trajectory count {len(rows)} != {expected_trajectories}")
    global_ids = [int(row["global_trajectory_id"]) for row in rows]
    if len(set(global_ids)) != len(global_ids):
        raise ValueError("Global trajectory IDs are not unique")
    counts = Counter(int(row["rollout_id"]) for row in rows)
    bad_counts = {key: value for key, value in counts.items() if value != samples_per_rollout}
    if bad_counts:
        raise ValueError(f"Unexpected per-rollout trajectory counts: {bad_counts}")

    windows = [
        summarize_rows(
            [row for row in rows if start <= int(row["rollout_id"]) <= end],
            start,
            end,
        )
        for start, end in MILESTONE_WINDOWS
    ]
    per_rollout = [
        summarize_rows(
            [row for row in rows if int(row["rollout_id"]) == rollout_id],
            rollout_id,
            rollout_id,
        )
        for rollout_id in rollout_ids
    ]
    first, last = windows[0], windows[-1]
    return {
        "schema_version": 1,
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "trajectories": len(rows),
            "rollouts": expected_rollouts,
            "samples_per_rollout": samples_per_rollout,
        },
        "window_definition": (
            "Four consecutive 23-rollout windows aligned with saved OPD milestones; "
            "each window contains 1,472 sampled trajectories."
        ),
        "milestone_windows": windows,
        "per_rollout": per_rollout,
        "first_to_last_window_change": {
            "mean_response_tokens": last["length"]["mean"] - first["length"]["mean"],
            "cap_hit_rate_percentage_points": percentage_point_change(first, last, "cap_hit_rate"),
            "natural_stop_rate_percentage_points": percentage_point_change(
                first, last, "natural_stop_rate"
            ),
            "think_closed_rate_percentage_points": percentage_point_change(
                first, last, "think_closed_rate"
            ),
            "eligible_final_answer_rate_percentage_points": percentage_point_change(
                first, last, "eligible_final_answer_rate"
            ),
            "im_end_rate_percentage_points": 100.0
            * (
                last["terminal_condition_rates"].get("im_end", 0.0)
                - first["terminal_condition_rates"].get("im_end", 0.0)
            ),
            "repetition_indicator_rate_percentage_points": percentage_point_change(
                first, last, "repetition_indicator_rate"
            ),
            "sampled_reverse_kl": float(last["sampled_reverse_kl"])
            - float(first["sampled_reverse_kl"]),
        },
        "interpretation": {
            "correctness": "Not measured: OpenThoughts rollout responses are not verified gold answers.",
            "causality": (
                "Windows contain different prompts, so changes are descriptive training-time trends, "
                "not a controlled within-prompt causal estimate."
            ),
            "stopping": (
                "The 16K cutoff bounded compute and supplied no synthetic terminal target. "
                "Think closure and natural stopping therefore report observed behavior, not an explicit length reward."
            ),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# OPD learning dynamics",
        "",
        "Each row is a consecutive 23-rollout window containing 1,472 trajectories.",
        "",
        "| Rollouts | Mean tokens | Cap hit | Think closed | Eligible final | `<|im_end|>` | Repetition | Sampled reverse-KL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["milestone_windows"]:
        lines.append(
            f"| {row['rollout_start']}-{row['rollout_end']} | {row['length']['mean']:.1f} | "
            f"{row['cap_hit_rate']:.2%} | {row['think_closed_rate']:.2%} | "
            f"{row['eligible_final_answer_rate']:.2%} | "
            f"{row['terminal_condition_rates'].get('im_end', 0.0):.2%} | "
            f"{row['repetition_indicator_rate']:.2%} | {row['sampled_reverse_kl']:.4f} |"
        )
    delta = summary["first_to_last_window_change"]
    lines.extend(
        [
            "",
            "From the first to final window:",
            "",
            f"- Think closure changed by {delta['think_closed_rate_percentage_points']:+.2f} points.",
            f"- Natural stopping changed by {delta['natural_stop_rate_percentage_points']:+.2f} points.",
            f"- `<|im_end|>` termination changed by {delta['im_end_rate_percentage_points']:+.2f} points.",
            f"- Cap hits changed by {delta['cap_hit_rate_percentage_points']:+.2f} points.",
            f"- Mean response length changed by {delta['mean_response_tokens']:+.1f} tokens.",
            f"- Eligible final answers changed by {delta['eligible_final_answer_rate_percentage_points']:+.2f} points.",
            f"- Repetition indicators changed by {delta['repetition_indicator_rate_percentage_points']:+.2f} points.",
            f"- Sampled reverse-KL changed by {delta['sampled_reverse_kl']:+.4f}.",
            "",
            summary["interpretation"]["correctness"],
            summary["interpretation"]["causality"],
            summary["interpretation"]["stopping"],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.trajectory_jsonl.read_text().splitlines() if line]
    summary = build_summary(
        rows,
        source_path=args.trajectory_jsonl,
        expected_rollouts=args.expected_rollouts,
        expected_trajectories=args.expected_trajectories,
        samples_per_rollout=args.samples_per_rollout,
    )
    atomic_text(args.out_json, json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(args.out_md, render_markdown(summary))
    print(json.dumps(summary["first_to_last_window_change"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
