#!/usr/bin/env python3
"""Build the resumable strict-English OpenThoughts3 corpus and math splits."""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import math
import multiprocessing as mp
import os
import random
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from examples.qwen3_8b_opd_tillicum import _strict_english_filters as filters


REFERENCE_COUNTS = {
    "incomplete_think": 749_380,
    "non_english_or_mixed": 397_642,
    "valid": 332_843,
}


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=env.get("OT3_DATASET", "open-thoughts/OpenThoughts3-1.2M"))
    parser.add_argument("--split", default=env.get("OT3_SPLIT", "train"))
    parser.add_argument("--output-dir", default=env.get("STRICT_EN_DATASET_DIR"))
    parser.add_argument("--sft-out", default=env.get("SFT_PARQUET"))
    parser.add_argument("--opd-reserve-out", default=env.get("CLEANED_OPD_RESERVE_JSONL"))
    parser.add_argument("--opd-1k-out", default=env.get("CLEANED_OPD_1K_JSONL"))
    parser.add_argument("--opd-rest-out", default=env.get("CLEANED_OPD_4K_JSONL"))
    parser.add_argument("--metadata-out", default=env.get("SPLIT_METADATA"))
    parser.add_argument("--seed", type=int, default=int(env.get("DATA_SEED", "1234")))
    parser.add_argument("--shard-size", type=int, default=int(env.get("STRICT_EN_SHARD_SIZE", "10000")))
    parser.add_argument("--max-selected", type=int, default=int(env.get("STRICT_MATH_MAX_SELECTED", "50000")))
    parser.add_argument("--opd-reserve-size", type=int, default=int(env.get("STRICT_OPD_RESERVE_SIZE", "5000")))
    parser.add_argument("--sft-batch-size", type=int, default=int(env.get("SFT_ROLLOUT_BATCH_SIZE", "250")))
    parser.add_argument("--opd-batch-size", type=int, default=int(env.get("OPD_ROLLOUT_BATCH_SIZE", "128")))
    parser.add_argument("--opd-first-size", type=int, default=int(env.get("CLEANED_OPD_FIRST_SIZE", "1024")))
    parser.add_argument("--audit-per-shard", type=int, default=8)
    parser.add_argument("--workers", type=int, default=int(env.get("STRICT_EN_WORKERS", "8")))
    parser.add_argument("--max-source-rows", type=int)
    parser.add_argument("--skip-reference-gate", action="store_true")
    parser.add_argument("--force-finalize", action="store_true")
    args = parser.parse_args()
    required = ("output_dir", "sft_out", "opd_reserve_out", "opd_1k_out", "opd_rest_out", "metadata_out")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("missing output path(s): " + ", ".join(missing))
    return args


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    temp.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp = Path(handle.name)
    temp.replace(path)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    value = row.get("messages", row.get("conversations", row.get("conversation")))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    messages: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            raw_role = str(item.get("role", item.get("from", item.get("speaker", "")))).lower()
            role = {"human": "user", "gpt": "assistant", "model": "assistant"}.get(raw_role, raw_role)
            content = item.get("content", item.get("value", item.get("text", "")))
            if role in {"system", "user", "assistant", "tool"}:
                messages.append({"role": role, "content": "" if content is None else str(content)})
    if messages:
        return messages
    prompt = row.get("prompt", row.get("question", row.get("problem")))
    response = row.get("response", row.get("answer", row.get("solution")))
    if prompt is not None and response is not None:
        return [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}]
    return []


def split_sections(messages: list[dict[str, str]]) -> tuple[str, str, str, str]:
    prompt = "\n".join(m["content"] for m in messages if m["role"] == "user")
    assistant = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    if assistant.count("<think>") == 1 and assistant.count("</think>") == 1:
        prefix, remainder = assistant.split("<think>", 1)
        thinking, answer = remainder.split("</think>", 1)
        if prefix.strip():
            thinking = prefix + "\n" + thinking
    else:
        thinking, answer = "", ""
    return prompt, assistant, thinking, answer


def source_record(row: dict[str, Any], source_row_id: int, messages: list[dict[str, str]]) -> dict[str, Any]:
    prompt, assistant, _, _ = split_sections(messages)
    return {
        "source_row_id": source_row_id,
        "source_dataset": "open-thoughts/OpenThoughts3-1.2M",
        "source": row.get("source"),
        "domain": row.get("domain"),
        "difficulty": row.get("difficulty"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "assistant_sha256": hashlib.sha256(assistant.encode()).hexdigest(),
        "messages": messages,
    }


def evaluate_row(row: dict[str, Any], source_row_id: int, detector: Any) -> dict[str, Any]:
    messages = normalize_messages(row)
    malformed = not messages or not any(m["role"] == "user" for m in messages) or not any(
        m["role"] == "assistant" for m in messages
    )
    prompt, assistant, thinking, answer = split_sections(messages)
    complete_think = assistant.find("<think>") >= 0 and assistant.find("</think>") > assistant.find("<think>")
    exact_think = assistant.count("<think>") == 1 and assistant.count("</think>") == 1
    sections = {"prompt": prompt, "thinking": thinking if exact_think else assistant, "answer": answer}
    language_reasons: list[str] = []
    for name, text in sections.items():
        language_reasons.extend(f"{name}:{reason}" for reason in filters.english_rejection_reasons(text, detector))
    noise_reasons = filters.noise_rejection_reasons("\n".join(m["content"] for m in messages))
    accepted = not malformed and complete_think and not language_reasons and not noise_reasons
    strict_math_reasons: list[str] = []
    if accepted:
        if str(row.get("domain", "")).lower() != "math":
            strict_math_reasons.append("domain_not_math")
        if not exact_think:
            strict_math_reasons.append("not_exactly_one_think_pair")
        if not answer.strip():
            strict_math_reasons.append("missing_post_think_answer")
        if not filters.is_math_heavy(prompt, thinking, answer):
            strict_math_reasons.append("not_math_heavy")
        strict_math_reasons.extend(filters.repetition_rejection_reasons(thinking))
    record = source_record(row, source_row_id, messages) if accepted else None
    return {
        "source_row_id": source_row_id,
        "malformed": malformed,
        "incomplete_think": not complete_think,
        "language_reasons": sorted(set(language_reasons)),
        "noise_reasons": sorted(set(noise_reasons)),
        "accepted": accepted,
        "strict_math": accepted and not strict_math_reasons,
        "strict_math_reasons": sorted(set(strict_math_reasons)),
        "record": record,
    }


_WORKER_DETECTOR: Any | None = None


def init_worker() -> None:
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = filters.build_detector()


def evaluate_task(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    if _WORKER_DETECTOR is None:
        raise RuntimeError("strict-English worker detector was not initialized")
    row, source_row_id = task
    return evaluate_row(row, source_row_id, _WORKER_DETECTOR)


def write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temp, compression="zstd")
    temp.replace(path)


def process_shard(ds: Any, start: int, end: int, output_dir: Path, pool: Any, audit_n: int) -> None:
    shard_name = f"part-{start:07d}-{end - 1:07d}"
    stats_path = output_dir / "shard_stats" / f"{shard_name}.json"
    success_path = output_dir / "shard_success" / f"{shard_name}._SUCCESS"
    if stats_path.exists() and success_path.exists():
        print(f"Skipping completed shard {shard_name}", flush=True)
        return

    accepted: list[dict[str, Any]] = []
    math_rows: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    audit: dict[str, list[dict[str, Any]]] = {"accepted": [], "rejected": []}
    tasks = ((dict(ds[idx]), idx) for idx in range(start, end))
    results = pool.imap(evaluate_task, tasks, chunksize=8)
    for result in results:
        counts["source_rows"] += 1
        counts["incomplete_think"] += int(result["incomplete_think"])
        counts["malformed"] += int(result["malformed"])
        counts["non_english_or_mixed"] += int(bool(result["language_reasons"]))
        counts["noise"] += int(bool(result["noise_reasons"]))
        counts["valid"] += int(result["accepted"])
        counts["strict_math_candidates"] += int(result["strict_math"])
        for reason in result["language_reasons"] + result["noise_reasons"] + result["strict_math_reasons"]:
            reasons[reason] += 1
        if result["accepted"]:
            accepted.append(result["record"])
            if result["strict_math"]:
                math_rows.append(result["record"])
        bucket = "accepted" if result["accepted"] else "rejected"
        if len(audit[bucket]) < audit_n:
            audit[bucket].append({key: result[key] for key in result if key != "record"})

    write_parquet_atomic(output_dir / "parquet" / f"{shard_name}.parquet", accepted)
    write_jsonl_atomic(output_dir / "math_candidates" / f"{shard_name}.jsonl", math_rows)
    atomic_text(
        stats_path,
        json.dumps(
            {"start": start, "end": end, "counts": dict(counts), "reasons": dict(reasons), "audit": audit},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    atomic_text(success_path, "ok\n")
    print(f"Completed {shard_name}: valid={len(accepted)} strict_math={len(math_rows)}", flush=True)


def aggregate_shards(output_dir: Path) -> tuple[collections.Counter[str], collections.Counter[str]]:
    counts: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    for path in sorted((output_dir / "shard_stats").glob("part-*.json")):
        data = json.loads(path.read_text())
        counts.update(data["counts"])
        reasons.update(data["reasons"])
    return counts, reasons


def reference_gate(counts: dict[str, int], full_run: bool) -> dict[str, Any]:
    checks = {
        "incomplete_within_1pct": abs(counts["incomplete_think"] - REFERENCE_COUNTS["incomplete_think"])
        <= math.ceil(REFERENCE_COUNTS["incomplete_think"] * 0.01),
        "language_at_least_95pct_reference": counts["non_english_or_mixed"] >= 377_760,
        "valid_at_most_105pct_reference": counts["valid"] <= 349_486,
        "at_least_2048_strict_math": counts["strict_math_candidates"] >= 2_048,
    }
    return {"required": full_run, "passed": all(checks.values()) if full_run else True, "checks": checks}


def read_jsonl_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def with_split_metadata(row: dict[str, Any], split: str, offset: int, seed: int) -> dict[str, Any]:
    metadata = {
        "source_row_id": row["source_row_id"],
        "source": row.get("source"),
        "domain": row.get("domain"),
        "difficulty": row.get("difficulty"),
        "prompt_sha256": row["prompt_sha256"],
        "assistant_sha256": row["assistant_sha256"],
        "strict_en_split": split,
        "strict_en_offset": offset,
        "strict_en_seed": seed,
    }
    if split == "sft":
        return {"messages": row["messages"], "metadata": metadata}
    prompt = "\n".join(m["content"] for m in row["messages"] if m["role"] == "user")
    return {"prompt": prompt, "metadata": metadata}


def finalize(args: argparse.Namespace, output_dir: Path, source_count: int) -> None:
    counts, reasons = aggregate_shards(output_dir)
    full_run = args.max_source_rows is None and counts["source_rows"] == source_count
    gate = reference_gate(counts, full_run and not args.skip_reference_gate)
    if not gate["passed"] and not args.force_finalize:
        raise SystemExit("Strict-English count preflight failed:\n" + json.dumps({"counts": counts, "gate": gate}, indent=2))

    candidates = read_jsonl_paths(sorted((output_dir / "math_candidates").glob("part-*.jsonl")))
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: min(len(candidates), args.max_selected)]
    if len(selected) < 2_048:
        raise SystemExit(f"Need at least 2,048 strict math rows, found {len(selected)}")

    if len(selected) >= 10_000:
        opd_reserved_count = min(args.opd_reserve_size, len(selected))
    else:
        opd_reserved_count = len(selected) // 2
    sft_raw = selected[:-opd_reserved_count]
    opd_reserved = selected[-opd_reserved_count:]
    sft_effective_count = (len(sft_raw) // args.sft_batch_size) * args.sft_batch_size
    opd_effective_count = (len(opd_reserved) // args.opd_batch_size) * args.opd_batch_size
    if opd_effective_count < args.opd_first_size:
        raise SystemExit(f"OPD effective count {opd_effective_count} is below first phase {args.opd_first_size}")
    sft_rows = [with_split_metadata(row, "sft", i, args.seed) for i, row in enumerate(sft_raw[:sft_effective_count])]
    opd_rows = [with_split_metadata(row, "opd", i, args.seed) for i, row in enumerate(opd_reserved)]
    opd_effective = opd_rows[:opd_effective_count]
    opd_first = opd_effective[: args.opd_first_size]
    opd_rest = opd_effective[args.opd_first_size :]

    sft_ids = {row["metadata"]["source_row_id"] for row in sft_rows}
    opd_ids = {row["metadata"]["source_row_id"] for row in opd_rows}
    sft_hashes = {row["metadata"]["prompt_sha256"] for row in sft_rows}
    opd_hashes = {row["metadata"]["prompt_sha256"] for row in opd_rows}
    if sft_ids & opd_ids or sft_hashes & opd_hashes:
        raise RuntimeError("Strict SFT/OPD source-id or prompt-hash overlap detected")

    write_jsonl_atomic(Path(args.sft_out), sft_rows)
    write_jsonl_atomic(Path(args.opd_reserve_out), opd_rows)
    write_jsonl_atomic(Path(args.opd_1k_out), opd_first)
    write_jsonl_atomic(Path(args.opd_rest_out), opd_rest)

    metadata = {
        "schema_version": 1,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "detector": {"package": "lingua-language-detector", "version": "2.2.0", "mode": "all_languages_low_accuracy"},
        "reference_counts": REFERENCE_COUNTS,
        "counts": dict(counts),
        "reference_gate": gate,
        "reason_frequencies": dict(reasons.most_common()),
        "selection": {
            "candidate_count": len(candidates),
            "selected_total": len(selected),
            "sft_raw": len(sft_raw),
            "sft_effective": len(sft_rows),
            "sft_batch_leftovers": len(sft_raw) - len(sft_rows),
            "opd_reserved": len(opd_rows),
            "opd_effective": len(opd_effective),
            "opd_batch_leftovers": len(opd_rows) - len(opd_effective),
            "opd_first": len(opd_first),
            "opd_continuation": len(opd_rest),
            "method": "seeded_shuffle_strict_math_then_tail_opd_reserve",
        },
        "disjointness": {
            "source_id_intersection": 0,
            "prompt_hash_intersection": 0,
            "sft_source_ids_sha256": sha256_json(sorted(sft_ids)),
            "opd_source_ids_sha256": sha256_json(sorted(opd_ids)),
        },
        "outputs": {
            "overall_parquet_dir": str(output_dir / "parquet"),
            "sft": args.sft_out,
            "opd_reserve": args.opd_reserve_out,
            "opd_first": args.opd_1k_out,
            "opd_continuation": args.opd_rest_out,
        },
    }
    atomic_text(Path(args.metadata_out), json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    atomic_text(output_dir / "reason_frequencies.json", json.dumps(dict(reasons.most_common()), indent=2) + "\n")
    atomic_text(output_dir / "_SUCCESS", json.dumps({"metadata": args.metadata_out, "counts": dict(counts)}) + "\n")
    print(json.dumps({"counts": dict(counts), "gate": gate, "selection": metadata["selection"]}, indent=2), flush=True)


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.dataset} split={args.split}", flush=True)
    ds = load_dataset(args.dataset, split=args.split, cache_dir=os.environ.get("HF_HOME"))
    source_count = len(ds)
    if args.max_source_rows is not None:
        source_count = min(source_count, args.max_source_rows)

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=init_worker) as pool:
        for start in range(0, source_count, args.shard_size):
            process_shard(ds, start, min(start + args.shard_size, source_count), output_dir, pool, args.audit_per_shard)
    finalize(args, output_dir, source_count)


if __name__ == "__main__":
    main()
