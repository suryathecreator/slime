#!/usr/bin/env python3
"""Record the abandoned 32K replay attempt without publishing a proxy score."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any


SHARD_RE = re.compile(
    r"CAP_PROXY_SHARD stage=(?P<stage>\S+) source=(?P<source>\S+) "
    r"shard=(?P<shard>\d+)/(?P<num_shards>\d+) selected=(?P<selected>\d+) "
    r"pending=(?P<pending>\d+) policy=(?P<policy>[0-9a-f]+)"
)
TOKEN_MISMATCH_RE = re.compile(
    r"Cap proxy prefix mismatch at index (?P<index>\d+), token offset (?P<offset>\d+): "
    r"source=(?P<source_token>\d+) rerun=(?P<rerun_token>\d+)"
)
LENGTH_MISMATCH_RE = re.compile(
    r"Cap proxy prefix mismatch at index (?P<index>\d+): rerun has (?P<rerun_length>\d+) "
    r"tokens, source has (?P<source_length>\d+)"
)
COMPLETED_RE = re.compile(r"Processed prompts:\s+100%.*?\|\s*(?P<done>\d+)/(?P<total>\d+)\s")
AGGREGATE_RE = re.compile(
    r"Proxy must contain exactly the source cap-hit rows; expected=(?P<expected>\d+) "
    r"found=(?P<found>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument(
        "--cancelled-job",
        action="append",
        default=[],
        metavar="JOB_ID:STAGE",
        help="Downstream job cancelled after the attempt was abandoned.",
    )
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


def cancelled_jobs(values: list[str]) -> list[dict[str, Any]]:
    rows = []
    for value in values:
        job_id, separator, stage = value.partition(":")
        if not separator or not job_id.isdigit() or not stage:
            raise ValueError(f"Invalid --cancelled-job value: {value}")
        rows.append({"job_id": int(job_id), "stage": stage, "state": "CANCELLED"})
    return rows


def build_attempt(
    *,
    log_path: Path,
    source_summary_path: Path,
    job_id: int,
    cancelled: list[dict[str, Any]],
) -> dict[str, Any]:
    text = log_path.read_text(errors="replace")
    source_summary = json.loads(source_summary_path.read_text())
    shards = [
        {
            "stage": match.group("stage"),
            "source_stage": match.group("source"),
            "shard_index": int(match.group("shard")),
            "num_shards": int(match.group("num_shards")),
            "selected": int(match.group("selected")),
            "pending": int(match.group("pending")),
            "policy_sha256": match.group("policy"),
        }
        for match in SHARD_RE.finditer(text)
    ]
    if not shards:
        raise ValueError("No cap-proxy shard records found")
    shards.sort(key=lambda row: row["shard_index"])
    selected = sum(row["selected"] for row in shards)
    observed_completed_sizes = sorted(
        {int(match.group("total")) for match in COMPLETED_RE.finditer(text) if match.group("done") == match.group("total")}
    )
    selected_sizes = sorted(row["selected"] for row in shards)
    all_shards_generated = observed_completed_sizes == selected_sizes
    mismatches: list[dict[str, Any]] = []
    for match in TOKEN_MISMATCH_RE.finditer(text):
        mismatches.append(
            {
                "kind": "token",
                "eval_index": int(match.group("index")),
                "first_mismatch_offset": int(match.group("offset")),
                "source_token_id": int(match.group("source_token")),
                "rerun_token_id": int(match.group("rerun_token")),
            }
        )
    for match in LENGTH_MISMATCH_RE.finditer(text):
        mismatches.append(
            {
                "kind": "length",
                "eval_index": int(match.group("index")),
                "source_generated_tokens": int(match.group("source_length")),
                "rerun_generated_tokens": int(match.group("rerun_length")),
            }
        )
    mismatches.sort(key=lambda row: row["eval_index"])
    aggregate = AGGREGATE_RE.search(text)
    if aggregate is None:
        raise ValueError("Missing fail-closed aggregate record")
    expected = int(aggregate.group("expected"))
    published = int(aggregate.group("found"))
    if selected != expected or int(source_summary["cap_hits"]) != expected:
        raise ValueError(
            f"Selected/source/aggregate cap counts disagree: {selected}/{source_summary['cap_hits']}/{expected}"
        )
    return {
        "schema_version": 1,
        "status": "abandoned_incomplete_no_score",
        "job": {"job_id": job_id, "stage": shards[0]["stage"], "state": "FAILED"},
        "source_evaluation": {
            "stage": source_summary["stage"],
            "summary_path": str(source_summary_path),
            "summary_sha256": sha256_file(source_summary_path),
            "correct": int(source_summary["correct"]),
            "total": int(source_summary["total"]),
            "cap_hits": int(source_summary["cap_hits"]),
        },
        "attempt": {
            "target_total_context_tokens": 32768,
            "selected_cap_hit_rows": selected,
            "shards": shards,
            "observed_completed_shard_sizes": observed_completed_sizes,
            "all_selected_generation_attempts_reached_engine_completion": all_shards_generated,
            "generation_attempts_reached_engine_completion": selected if all_shards_generated else None,
            "published_rows": published,
            "score_published": False,
            "saved_partial_generation_rows": 0,
            "prefix_mismatches_observed": mismatches,
            "fail_closed": published == 0,
        },
        "log": {
            "path": str(log_path),
            "sha256": sha256_file(log_path),
            "bytes": log_path.stat().st_size,
        },
        "fidelity_decision": {
            "exact_replay_unreliable": (
                "A seeded stochastic rerun is not guaranteed to reproduce an earlier sampled prefix. "
                "Batch scheduling, kernel and floating-point paths, RNG consumption, engine version, and "
                "the requested generation budget can all change token sampling."
            ),
            "prefix_continuation_rejected": (
                "Continuing from the saved 16K prefix would condition on an old trajectory while restarting "
                "the sampler state. It is not the same experiment as drawing the response from the original "
                "prompt with a 32K budget, so it is not reported as a fidelity-preserving evaluation."
            ),
            "required_for_valid_32k_ablation": (
                "Regenerate all 500 prompts from their original inputs under one pinned 32K protocol. "
                "Context length alone would still not perfectly reproduce an external result unless the "
                "checkpoint, prompt template, tokenizer, stop rules, sampler implementation, seeds, scorer, "
                "engine/runtime, batching, and hardware behavior are also aligned."
            ),
            "future_work": (
                "A fresh 32K evaluation can be run later as an explicit ablation, alongside other protocol "
                "ablations, without changing the completed fixed-16K primary result."
            ),
        },
        "cancelled_downstream_jobs": cancelled,
    }


def render_markdown(value: dict[str, Any]) -> str:
    attempt = value["attempt"]
    lines = [
        "# Abandoned 32K context replay attempt",
        "",
        f"- Job: `{value['job']['job_id']}` ({value['job']['state']})",
        f"- Source cap-hit rows selected: {attempt['selected_cap_hit_rows']}",
        f"- Generation attempts reaching engine completion: {attempt['generation_attempts_reached_engine_completion']}",
        f"- Saved/published rows: {attempt['saved_partial_generation_rows']}/{attempt['published_rows']}",
        "- Score published: no",
        "",
        "Each shard finished generating its assigned attempts, but the fail-closed exact-prefix check failed on the first checked row. Examples:",
        "",
    ]
    for row in attempt["prefix_mismatches_observed"]:
        if row["kind"] == "token":
            lines.append(
                f"- Index {row['eval_index']}: first mismatch at token {row['first_mismatch_offset']} "
                f"(`{row['source_token_id']}` versus `{row['rerun_token_id']}`)."
            )
        else:
            lines.append(
                f"- Index {row['eval_index']}: rerun stopped at {row['rerun_generated_tokens']} tokens "
                f"before the saved {row['source_generated_tokens']}-token prefix."
            )
    decision = value["fidelity_decision"]
    lines.extend(
        [
            "",
            "## Fidelity decision",
            "",
            decision["exact_replay_unreliable"],
            "",
            decision["prefix_continuation_rejected"],
            "",
            decision["required_for_valid_32k_ablation"],
            "",
            decision["future_work"],
            "",
            "## Cancelled downstream jobs",
            "",
        ]
    )
    for row in value["cancelled_downstream_jobs"]:
        lines.append(f"- `{row['job_id']}`: {row['stage']} ({row['state']})")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    value = build_attempt(
        log_path=args.log,
        source_summary_path=args.source_summary,
        job_id=args.job_id,
        cancelled=cancelled_jobs(args.cancelled_job),
    )
    atomic_text(args.out_json, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(args.out_md, render_markdown(value))
    print(json.dumps(value["attempt"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
