#!/usr/bin/env python3
"""Audit preserved OPD trajectory tensors without changing the training loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from examples.qwen3_8b_32b_full_repro.math500_scorer import apply_discourse, extract_events


TERMINALS = {151643: "eos", 151645: "im_end"}
DOUBT = re.compile(r"(?i)\b(?:maybe|perhaps|possibly|not\s+sure|uncertain|doubt)\b")
RETRACTION = re.compile(r"(?i)\b(?:wait|actually|scratch\s+that|wrong|incorrect|retract)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=Path(os.environ["OPD_ROLLOUT_LOG_DIR"]))
    parser.add_argument("--out-jsonl", type=Path, default=Path(os.environ["OPD_ROLLOUT_AUDIT_JSONL"]))
    parser.add_argument("--out-json", type=Path, default=Path(os.environ["OPD_ROLLOUT_AUDIT_JSON"]))
    parser.add_argument("--out-md", type=Path, default=Path(os.environ["OPD_ROLLOUT_AUDIT_MD"]))
    parser.add_argument("--expected-rollouts", type=int, default=int(os.environ.get("OPD_NUM_ROLLOUT", "92")))
    parser.add_argument("--expected-trajectories", type=int, default=5888)
    parser.add_argument("--max-files", type=int)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    temp.replace(path)


def fake_parse(value: str, **_kwargs: Any) -> list[str]:
    return [value] if value.strip() else []


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def teacher_tail(sample: dict[str, Any], response_length: int) -> list[float]:
    values = ((sample.get("reward") or {}).get("meta_info") or {}).get("input_token_logprobs") or []
    logprobs = [float(item[0]) for item in values[1:] if item and item[0] is not None]
    return logprobs[-response_length:] if response_length else []


def repetition_indicator(text: str) -> bool:
    tail = text[-8000:]
    lines = [line.strip() for line in tail.splitlines() if len(line.strip()) >= 20]
    if lines and max(Counter(lines).values()) >= 4:
        return True
    chunks = [tail[index : index + 80] for index in range(0, max(0, len(tail) - 79), 80)]
    return bool(chunks and max(Counter(chunks).values()) >= 4)


def audit_sample(root_rollout_id: int, local_order: int, sample: dict[str, Any]) -> dict[str, Any]:
    response_length = int(sample.get("response_length") or 0)
    tokens = [int(value) for value in (sample.get("tokens") or [])]
    response_tokens = tokens[-response_length:] if response_length else []
    student = [float(value) for value in (sample.get("rollout_log_probs") or [])][-response_length:]
    teacher = teacher_tail(sample, response_length)
    aligned = min(len(student), len(teacher), response_length)
    student = student[-aligned:] if aligned else []
    teacher = teacher[-aligned:] if aligned else []
    terminal_id = response_tokens[-1] if response_tokens and response_tokens[-1] in TERMINALS else None
    text = str(sample.get("response") or "")
    events, malformed = extract_events(text, 0, parse_fn=fake_parse)
    selected, selection_history = apply_discourse(text, events)
    metadata = sample.get("metadata") or {}
    reverse_kl = [s - t for s, t in zip(student, teacher, strict=True)]
    global_id = root_rollout_id * int(os.environ.get("OPD_GLOBAL_BATCH_SIZE", "64")) + local_order
    return {
        "global_trajectory_id": global_id,
        "rollout_id": root_rollout_id,
        "local_order_within_rollout": local_order,
        "sample_index_reported": sample.get("index"),
        "group_index_reported": sample.get("group_index"),
        "source_row_id": metadata.get("source_row_id"),
        "prompt_sha256": metadata.get("prompt_sha256"),
        "prompt_source": metadata.get("source"),
        "prompt_domain": metadata.get("domain"),
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "response_tokens": response_length,
        "status": str(sample.get("status")),
        "cap_hit": str(sample.get("status")).lower() == "truncated" or response_length >= int(os.environ["OPD_MAX_RESPONSE_LEN"]),
        "think_closed": "</think>" in text,
        "eligible_final_answer": selected.candidate if selected else None,
        "answer_events": len(events),
        "answer_replacements": sum(event.superseded for event in events),
        "doubt": bool(DOUBT.search(text)) or any(event.doubtful for event in events),
        "retraction": bool(RETRACTION.search(text)) or any(event.retracted for event in events),
        "malformed_boxes": malformed,
        "repetition_indicator": repetition_indicator(text),
        "terminal_token_id": terminal_id,
        "terminal_condition": TERMINALS.get(terminal_id),
        "teacher_terminal_logprob": teacher[-1] if terminal_id is not None and teacher else None,
        "student_terminal_logprob": student[-1] if terminal_id is not None and student else None,
        "teacher_minus_student_sampled_logprob": statistics.fmean(t - s for s, t in zip(student, teacher, strict=True)) if aligned else None,
        "sampled_reverse_kl": statistics.fmean(reverse_kl) if reverse_kl else None,
        "aligned_logprob_tokens": aligned,
        "selection_history": selection_history,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(
        (path for path in args.rollout_dir.glob("rollout_*.pt") if ".failed_" not in path.name),
        key=lambda path: int(re.search(r"rollout_(\d+)", path.name).group(1)),
    )
    if args.max_files is not None:
        paths = paths[: args.max_files]
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    rollout_ids: list[int] = []
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        rollout_id = int(artifact["rollout_id"])
        rollout_ids.append(rollout_id)
        for local_order, sample in enumerate(artifact["samples"]):
            row = audit_sample(rollout_id, local_order, sample)
            global_id = int(row["global_trajectory_id"])
            if global_id in seen:
                raise SystemExit(f"Duplicate global trajectory ID {global_id}")
            seen.add(global_id)
            rows.append(row)
    if args.max_files is None:
        if rollout_ids != list(range(args.expected_rollouts)):
            raise SystemExit(f"Rollout IDs are not globally complete: {rollout_ids}")
        if len(rows) != args.expected_trajectories:
            raise SystemExit(f"Trajectory count {len(rows)} != {args.expected_trajectories}")
    if not rows:
        raise SystemExit(f"No rollout trajectories found under {args.rollout_dir}")
    lengths = [int(row["response_tokens"]) for row in rows]

    def numeric(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row[key] is not None]

    teacher_terminal = numeric("teacher_terminal_logprob")
    student_terminal = numeric("student_terminal_logprob")
    summary = {
        "schema_version": 1,
        "rollout_files": len(paths),
        "trajectories": len(rows),
        "globally_unique_trajectory_ids": len(seen) == len(rows),
        "length": {"mean": statistics.fmean(lengths), "median": statistics.median(lengths), "p90": percentile(lengths, 0.90), "p95": percentile(lengths, 0.95), "max": max(lengths)},
        "cap_hits": sum(row["cap_hit"] for row in rows),
        "cap_hit_rate": sum(row["cap_hit"] for row in rows) / len(rows),
        "think_closed_rate": sum(row["think_closed"] for row in rows) / len(rows),
        "eligible_final_answer_rate": sum(row["eligible_final_answer"] is not None for row in rows) / len(rows),
        "terminal_conditions": dict(Counter(str(row["terminal_condition"]) for row in rows)),
        "answer_events": sum(int(row["answer_events"]) for row in rows),
        "answer_replacements": sum(int(row["answer_replacements"]) for row in rows),
        "doubts": sum(bool(row["doubt"]) for row in rows),
        "retractions": sum(bool(row["retraction"]) for row in rows),
        "repetition_indicators": sum(bool(row["repetition_indicator"]) for row in rows),
        "teacher_minus_student_sampled_logprob": statistics.fmean(numeric("teacher_minus_student_sampled_logprob")),
        "sampled_reverse_kl": statistics.fmean(numeric("sampled_reverse_kl")),
        "mean_teacher_terminal_logprob": statistics.fmean(teacher_terminal) if teacher_terminal else None,
        "mean_student_terminal_logprob": statistics.fmean(student_terminal) if student_terminal else None,
        "prompt_sources": dict(Counter(str(row["prompt_source"]) for row in rows)),
        "prompt_domains": dict(Counter(str(row["prompt_domain"]) for row in rows)),
        "behavior_metrics_fatal": False,
        "reference_correctness": "not computed: OpenThoughts responses are not verified gold references",
    }
    atomic_text(args.out_jsonl, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    atomic_text(args.out_json, json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(
        args.out_md,
        "# OPD rollout audit\n\n"
        f"- Trajectories: {summary['trajectories']}\n"
        f"- Mean/median/p95/max tokens: {summary['length']['mean']:.1f}/{summary['length']['median']}/{summary['length']['p95']}/{summary['length']['max']}\n"
        f"- Cap-hit rate: {summary['cap_hit_rate']:.2%}\n"
        f"- Think closure / eligible final answer: {summary['think_closed_rate']:.2%}/{summary['eligible_final_answer_rate']:.2%}\n"
        f"- Sampled reverse-KL: {summary['sampled_reverse_kl']:.6f}\n"
        f"- Mean teacher/student terminal log-probability: {summary['mean_teacher_terminal_logprob']}/{summary['mean_student_terminal_logprob']}\n\n"
        "All behavior thresholds were diagnostic and nonfatal. Capped and failed trajectories remain preserved.\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
