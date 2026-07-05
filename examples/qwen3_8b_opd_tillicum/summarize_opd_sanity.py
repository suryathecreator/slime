#!/usr/bin/env python3
"""Aggregate OPD rollout sanity JSON and train-log diagnostics."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ROLLOUT_RE = re.compile(r"rollout\s+(\d+):\s+(\{.*\})")
PERF_RE = re.compile(r"perf\s+(\d+):\s+(\{.*\})")
STEP_RE = re.compile(r"step\s+(\d+):\s+(\{.*\})")

CSV_COLUMNS = [
    "rollout_id",
    "sample_index_min",
    "sample_index_max",
    "n",
    "avg_response_tokens",
    "median_response_tokens",
    "min_response_tokens",
    "max_response_tokens",
    "cap_hit_rate",
    "completed_rate",
    "final_answer_rate",
    "rollout_log_probs",
    "actor_log_probs",
    "ref_log_probs",
    "teacher_log_probs",
    "opd_reverse_kl",
    "advantages",
    "train_rollout_logprob_abs_diff",
    "kl_loss",
    "pg_clipfrac",
    "ppo_kl",
    "grad_norm",
    "sanity_passed",
    "violations",
    "ref_load",
]

COLUMN_DESCRIPTIONS = {
    "rollout_id": "Rollout/update index, derived from sample range when Slime did not attach a rollout id.",
    "sample_index_min": "First OPD data row offset represented in the sanity batch.",
    "sample_index_max": "Last OPD data row offset represented in the sanity batch.",
    "n": "Number of samples in the rollout batch.",
    "avg_response_tokens": "Mean generated response length over samples.",
    "median_response_tokens": "Median generated response length over samples when present in the Slurm perf log.",
    "min_response_tokens": "Minimum generated response length over samples when present in the Slurm perf log.",
    "max_response_tokens": "Maximum generated response length over samples.",
    "cap_hit_rate": "Fraction of samples that hit the response cap or were marked truncated.",
    "completed_rate": "Fraction of samples marked completed.",
    "final_answer_rate": "Fraction of samples whose response tail contained a conservative final-answer marker.",
    "rollout_log_probs": "SGLang rollout-student logprobs, response-token-weighted over the rollout.",
    "actor_log_probs": "Megatron actor logprobs before update, response-token-weighted over the rollout.",
    "ref_log_probs": "Megatron reference logprobs, response-token-weighted over the rollout.",
    "teacher_log_probs": "Qwen3-32B teacher logprobs, response-token-weighted over the rollout.",
    "opd_reverse_kl": "Mean actor_log_probs - teacher_log_probs over response tokens.",
    "advantages": "Mean post-OPD advantage over response tokens.",
    "train_rollout_logprob_abs_diff": "Mean absolute difference between Megatron actor and SGLang rollout logprobs over response tokens.",
    "kl_loss": "Diagnostic reference-KL loss, response-token-weighted; coefficient is currently zero.",
    "pg_clipfrac": "Fraction of response tokens clipped by PPO-style policy loss.",
    "ppo_kl": "Mean old-vs-current actor KL used for policy-ratio diagnostics.",
    "grad_norm": "Megatron actor gradient norm after the rollout update.",
    "sanity_passed": "Whether the configured sanity thresholds were satisfied.",
    "violations": "Semicolon-separated threshold violations.",
    "ref_load": "Reference checkpoint path used for logged KL diagnostics.",
}

AVERAGING_NOTE = (
    "Averaging: logprob/KL/loss diagnostics are response-token-weighted rollout means. "
    "Response length statistics are sample statistics over the rollout. "
    "Cap/completion/final-answer rates are sample fractions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref-load", default="")
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--max-cap-hit-rate", type=float, default=1.0)
    parser.add_argument("--max-avg-response-tokens", type=float, default=math.inf)
    parser.add_argument("--min-final-answer-rate", type=float, default=0.0)
    return parser.parse_args()


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_literal_dict(raw: str) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def maybe_float(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    return ""


def sanity_rollout_id(summary: dict[str, Any], rollout_batch_size: int) -> int | None:
    rollout_id = summary.get("rollout_id")
    if rollout_id is not None:
        return int(rollout_id)
    sample_min = summary.get("sample_index_min")
    if sample_min is None:
        return None
    return int(sample_min) // rollout_batch_size


def load_sanity_rows(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not args.sanity_dir.exists():
        return rows

    for path in sorted(args.sanity_dir.glob("*.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rollout_id = sanity_rollout_id(summary, args.rollout_batch_size)
        if rollout_id is None:
            continue

        thresholds = summary.get("thresholds") or {
            "max_cap_hit_rate": args.max_cap_hit_rate,
            "max_avg_response_tokens": args.max_avg_response_tokens,
            "min_final_answer_rate": args.min_final_answer_rate,
        }
        violations = list(summary.get("violations") or [])
        if not violations:
            max_cap = float(thresholds.get("max_cap_hit_rate", args.max_cap_hit_rate))
            max_avg = float(thresholds.get("max_avg_response_tokens", args.max_avg_response_tokens))
            min_final = float(thresholds.get("min_final_answer_rate", args.min_final_answer_rate))
            if float(summary.get("cap_hit_rate", 0.0)) > max_cap:
                violations.append(f"cap_hit_rate {summary.get('cap_hit_rate'):.3f} > {max_cap:.3f}")
            if float(summary.get("avg_response_tokens", 0.0)) > max_avg:
                violations.append(f"avg_response_tokens {summary.get('avg_response_tokens'):.1f} > {max_avg:.1f}")
            if float(summary.get("final_answer_rate", 0.0)) < min_final:
                violations.append(f"final_answer_rate {summary.get('final_answer_rate'):.3f} < {min_final:.3f}")

        rows[rollout_id] = {
            "rollout_id": rollout_id,
            "sample_index_min": summary.get("sample_index_min", ""),
            "sample_index_max": summary.get("sample_index_max", ""),
            "n": summary.get("n", ""),
            "avg_response_tokens": maybe_float(summary.get("avg_response_tokens")),
            "max_response_tokens": maybe_float(summary.get("max_response_tokens")),
            "cap_hit_rate": maybe_float(summary.get("cap_hit_rate")),
            "completed_rate": maybe_float(summary.get("completed_rate")),
            "final_answer_rate": maybe_float(summary.get("final_answer_rate")),
            "sanity_passed": not violations,
            "violations": "; ".join(violations),
            "ref_load": args.ref_load,
        }
    return rows


def merge_log_metrics(rows: dict[int, dict[str, Any]], train_log: Path | None) -> None:
    if train_log is None or not train_log.exists():
        return

    with train_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = clean_line(line)
            for regex, kind in ((ROLLOUT_RE, "rollout"), (PERF_RE, "perf"), (STEP_RE, "step")):
                match = regex.search(line)
                if not match:
                    continue
                rollout_id = int(match.group(1))
                payload = parse_literal_dict(match.group(2))
                if payload is None:
                    continue
                row = rows.setdefault(rollout_id, {"rollout_id": rollout_id, "ref_load": ""})
                if kind == "rollout":
                    row.update(
                        {
                            "avg_response_tokens": payload.get(
                                "rollout/response_lengths", row.get("avg_response_tokens", "")
                            ),
                            "rollout_log_probs": payload.get("rollout/rollout_log_probs", ""),
                            "actor_log_probs": payload.get("rollout/log_probs", ""),
                            "ref_log_probs": payload.get("rollout/ref_log_probs", ""),
                            "teacher_log_probs": payload.get("rollout/teacher_log_probs", ""),
                            "opd_reverse_kl": payload.get("rollout/opd_reverse_kl", ""),
                            "advantages": payload.get("rollout/advantages", ""),
                        }
                    )
                elif kind == "perf":
                    row.update(
                        {
                            "avg_response_tokens": payload.get(
                                "rollout/response_len/mean", row.get("avg_response_tokens", "")
                            ),
                            "median_response_tokens": payload.get("rollout/response_len/median", ""),
                            "min_response_tokens": payload.get("rollout/response_len/min", ""),
                            "max_response_tokens": payload.get(
                                "rollout/response_len/max", row.get("max_response_tokens", "")
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "train_rollout_logprob_abs_diff": payload.get(
                                "train/train_rollout_logprob_abs_diff", ""
                            ),
                            "kl_loss": payload.get("train/kl_loss", ""),
                            "pg_clipfrac": payload.get("train/pg_clipfrac", ""),
                            "ppo_kl": payload.get("train/ppo_kl", ""),
                            "grad_norm": payload.get("train/grad_norm", ""),
                        }
                    )


def normalize_rows(rows: dict[int, dict[str, Any]], ref_load: str) -> list[dict[str, Any]]:
    normalized = []
    for rollout_id in sorted(rows):
        row = {column: rows[rollout_id].get(column, "") for column in CSV_COLUMNS}
        if not row["ref_load"]:
            row["ref_load"] = ref_load
        normalized.append(row)
    return normalized


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "opd_sanity_summary.json"
    csv_path = output_dir / "opd_sanity_summary.csv"
    md_path = output_dir / "opd_sanity_summary.md"

    json_path.write_text(json.dumps({"averaging_note": AVERAGING_NOTE, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OPD Sanity Summary",
        "",
        AVERAGING_NOTE,
        "",
        "| column | definition |",
        "| --- | --- |",
    ]
    for column in CSV_COLUMNS:
        lines.append(f"| `{column}` | {COLUMN_DESCRIPTIONS[column]} |")
    lines.extend(["", "| " + " | ".join(CSV_COLUMNS) + " |", "| " + " | ".join(["---"] * len(CSV_COLUMNS)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in CSV_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_sanity_rows(args)
    merge_log_metrics(rows, args.train_log)
    output_rows = normalize_rows(rows, args.ref_load)
    write_outputs(output_rows, args.output_dir)
    print(f"Wrote OPD sanity summary to {args.output_dir}")


if __name__ == "__main__":
    main()
