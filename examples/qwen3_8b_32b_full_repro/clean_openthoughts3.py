#!/usr/bin/env python3
"""Deterministic, resumable OpenThoughts3 cleanup and raw-row split."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable


REFERENCE = {
    "incomplete_think": 749_380,
    "assistant_contains_cjk": 397_642,
    "retained": 332_843,
}
RANGES = {
    "incomplete_think": (711_911, 786_849),
    "assistant_contains_cjk": (377_760, 417_524),
    "retained": (316_201, 349_485),
}
PROMPT_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
_OBVIOUS_TRUNCATION = re.compile(r"(?is)(?:\.{3,}|\b(?:to be continued|continued)\s*)$")

# Candidate blocks for Han, Hiragana, Katakana, Hangul, and Bopomofo letters.
# The general-category check below excludes punctuation, symbols, and unassigned
# code points inside these blocks.  The supplementary ranges include CJK
# extensions through Unicode 17; older Python UCDs safely classify newer,
# unknown code points as unassigned.
_CJK_CANDIDATE = re.compile(
    "["
    "\u1100-\u11ff"
    "\u3005-\u303b"
    "\u3040-\u30ff"
    "\u3100-\u318f"
    "\u31a0-\u31bf"
    "\u31f0-\u31ff"
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\ua960-\ua97f"
    "\uac00-\ud7ff"
    "\uf900-\ufaff"
    "\uff66-\uff9f"
    "\uffa0-\uffdc"
    "\U0001aff0-\U0001afff"
    "\U0001b000-\U0001b16f"
    "\U00020000-\U0002ee5f"
    "\U0002f800-\U0002fa1f"
    "\U00030000-\U0003347f"
    "]"
)


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=env.get("OT3_DATASET", "open-thoughts/OpenThoughts3-1.2M"))
    parser.add_argument("--revision", default=env.get("OT3_REVISION"), required=not bool(env.get("OT3_REVISION")))
    parser.add_argument("--split", default=env.get("OT3_SPLIT", "train"))
    parser.add_argument("--cache-dir", default=env.get("HF_HOME"))
    parser.add_argument("--tokenizer", default=env.get("STUDENT_HF_DIR"), required=not bool(env.get("STUDENT_HF_DIR")))
    parser.add_argument("--work-dir", default=env.get("CLEANUP_WORK_DIR"), required=not bool(env.get("CLEANUP_WORK_DIR")))
    parser.add_argument("--sft-out", default=env.get("SFT_PARQUET"), required=not bool(env.get("SFT_PARQUET")))
    parser.add_argument("--opd-reserve-out", default=env.get("CLEANED_OPD_RESERVE_JSONL"), required=not bool(env.get("CLEANED_OPD_RESERVE_JSONL")))
    parser.add_argument("--opd-headline-out", default=env.get("OPD_JSONL"), required=not bool(env.get("OPD_JSONL")))
    parser.add_argument("--audit-json", default=env.get("CLEANUP_AUDIT_JSON"), required=not bool(env.get("CLEANUP_AUDIT_JSON")))
    parser.add_argument("--audit-md", default=env.get("CLEANUP_AUDIT_MD"), required=not bool(env.get("CLEANUP_AUDIT_MD")))
    parser.add_argument("--split-metadata", default=env.get("SPLIT_METADATA"), required=not bool(env.get("SPLIT_METADATA")))
    parser.add_argument("--seed", type=int, default=int(env.get("DATA_SEED", "1234")))
    parser.add_argument("--sft-size", type=int, default=int(env.get("SFT_SIZE", "200000")))
    parser.add_argument("--opd-size", type=int, default=int(env.get("OPD_POOL_SIZE", "100000")))
    parser.add_argument("--headline-size", type=int, default=int(env.get("OPD_TRAIN_SIZE", "1472")))
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--audit-samples", type=int, default=64)
    parser.add_argument("--max-source-rows", type=int)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    temp.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp = Path(handle.name)
    temp.replace(path)


def atomic_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


MESSAGE_FIELDS = ("messages", "conversations", "conversation")
PROMPT_FIELDS = ("prompt", "problem", "question", "instruction", "input")
RESPONSE_FIELDS = ("response", "completion", "answer", "solution", "output")


def first_present(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str | None, Any | None]:
    for field in fields:
        if field in row and row[field] not in (None, ""):
            return field, row[field]
    return None, None


def message_source_field(row: dict[str, Any]) -> str:
    field, _ = first_present(row, MESSAGE_FIELDS)
    if field is not None:
        return field
    prompt_field, _ = first_present(row, PROMPT_FIELDS)
    response_field, _ = first_present(row, RESPONSE_FIELDS)
    if prompt_field is not None or response_field is not None:
        return f"flat:{prompt_field or 'missing'}+{response_field or 'missing'}"
    return "missing"


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    _, value = first_present(row, MESSAGE_FIELDS)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if isinstance(value, dict) and "messages" in value:
        value = value["messages"]
    messages: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", item.get("from", item.get("speaker", "")))).strip().lower()
            role = {"human": "user", "prompt": "user", "gpt": "assistant", "model": "assistant"}.get(role, role)
            content = item.get("content", item.get("value", item.get("text", "")))
            if role in {"system", "user", "assistant"}:
                messages.append({"role": role, "content": "" if content is None else str(content)})
    if messages:
        return messages
    _, prompt = first_present(row, PROMPT_FIELDS)
    _, response = first_present(row, RESPONSE_FIELDS)
    if prompt is not None:
        messages.append({"role": "user", "content": str(prompt)})
    if response is not None:
        messages.append({"role": "assistant", "content": str(response)})
    return messages


def prompt_and_response(messages: list[dict[str, str]]) -> tuple[str, str]:
    prompt = "\n".join(m["content"] for m in messages if m["role"] == "user")
    response = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    return prompt, response


def validate_think(response: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    opens = [m.start() for m in re.finditer(re.escape("<think>"), response)]
    closes = [m.start() for m in re.finditer(re.escape("</think>"), response)]
    if not opens:
        reasons.append("missing_open")
    if not closes:
        reasons.append("missing_close")
    if len(opens) != 1 or len(closes) != 1:
        reasons.append("duplicated_or_unbalanced_tags")
    if opens and closes and closes[0] < opens[0]:
        reasons.append("close_before_open")
    if len(opens) == 1 and len(closes) == 1 and closes[0] > opens[0]:
        thinking = response[opens[0] + len("<think>") : closes[0]]
        final = response[closes[0] + len("</think>") :]
        if not thinking.strip():
            reasons.append("empty_thinking")
        if not final.strip():
            reasons.append("empty_post_think")
        if _OBVIOUS_TRUNCATION.search(thinking) and not final.strip():
            reasons.append("obvious_truncation")
    return not reasons, sorted(set(reasons))


def first_cjk_letter(text: str) -> dict[str, Any] | None:
    """Return audit metadata for the first CJK-script Unicode letter."""
    for match in _CJK_CANDIDATE.finditer(text):
        character = match.group(0)
        if not unicodedata.category(character).startswith("L"):
            continue
        return {
            "index": match.start(),
            "character": character,
            "codepoint": f"U+{ord(character):04X}",
            "name": unicodedata.name(character, "UNNAMED"),
        }
    return None


def evaluate_task(task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    source_row_id, row = task
    messages = normalize_messages(row)
    prompt, response = prompt_and_response(messages)
    raw_chat_control_tokens = [token for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>") if token in response]
    malformed = not prompt.strip() or not response.strip() or bool(raw_chat_control_tokens)
    think_ok, think_reasons = validate_think(response)
    assistant_cjk_match = first_cjk_letter(response)
    assistant_contains_cjk = assistant_cjk_match is not None
    accepted = not malformed and think_ok and not assistant_contains_cjk
    record = None
    if accepted:
        record = {
            "source_row_id": source_row_id,
            "source": row.get("source"),
            "domain": row.get("domain"),
            "difficulty": row.get("difficulty"),
            "messages": messages,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        }
    return {
        "source_row_id": source_row_id,
        "message_source_field": message_source_field(row),
        "malformed": malformed,
        "raw_chat_control_tokens": raw_chat_control_tokens,
        "incomplete_think": not think_ok,
        "think_reasons": think_reasons,
        "assistant_contains_cjk": assistant_contains_cjk,
        "assistant_cjk_match": assistant_cjk_match,
        "retained": accepted,
        "record": record,
    }


def shard_paths(work_dir: Path, start: int, end: int) -> dict[str, Path]:
    name = f"part-{start:07d}-{end - 1:07d}"
    return {
        "stats": work_dir / "stats" / f"{name}.json",
        "classifications": work_dir / "classifications" / f"{name}.jsonl.gz",
        "accepted": work_dir / "accepted" / f"{name}.jsonl.gz",
        "success": work_dir / "success" / f"{name}._SUCCESS",
    }


def process_shard(dataset: Any, start: int, end: int, work_dir: Path, pool: Any, audit_samples: int) -> None:
    paths = shard_paths(work_dir, start, end)
    if all(paths[key].exists() for key in ("success", "stats", "classifications", "accepted")):
        print(f"CLEANUP_RESUME start={start} end={end}", flush=True)
        return
    results = pool.imap(evaluate_task, ((idx, dict(dataset[idx])) for idx in range(start, end)), chunksize=32)
    counts: collections.Counter[str] = collections.Counter()
    overlap: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    classifications: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, Any]]] = {"accepted": [], "rejected": []}
    for result in results:
        incomplete = bool(result["incomplete_think"])
        assistant_contains_cjk = bool(result["assistant_contains_cjk"])
        retained = bool(result["retained"])
        counts.update(
            source_rows=1,
            malformed=int(result["malformed"]),
            incomplete_think=int(incomplete),
            assistant_contains_cjk=int(assistant_contains_cjk),
            retained=int(retained),
        )
        overlap[f"incomplete_{int(incomplete)}__assistant_cjk_{int(assistant_contains_cjk)}"] += 1
        reasons.update(result["think_reasons"])
        reasons.update([f"assistant_cjk:{'present' if assistant_contains_cjk else 'absent'}"])
        reasons.update([f"raw_chat_control:{token}" for token in result["raw_chat_control_tokens"]])
        classification = {key: value for key, value in result.items() if key != "record"}
        classifications.append(classification)
        if retained:
            accepted.append(result["record"])
        bucket = "accepted" if retained else "rejected"
        if len(samples[bucket]) < audit_samples:
            samples[bucket].append(classification)
    atomic_gzip_jsonl(paths["classifications"], classifications)
    atomic_gzip_jsonl(paths["accepted"], accepted)
    atomic_text(paths["stats"], json.dumps({"start": start, "end": end, "counts": counts, "overlap": overlap, "reasons": reasons, "samples": samples}, indent=2, sort_keys=True) + "\n")
    atomic_text(paths["success"], "ok\n")
    print(f"CLEANUP_SHARD_COMPLETE start={start} end={end} retained={counts['retained']}", flush=True)


def aggregate(work_dir: Path) -> tuple[collections.Counter[str], collections.Counter[str], collections.Counter[str], dict[str, list[dict[str, Any]]]]:
    counts: collections.Counter[str] = collections.Counter()
    overlap: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[dict[str, Any]]] = {"accepted": [], "rejected": []}
    for path in sorted((work_dir / "stats").glob("part-*.json")):
        data = json.loads(path.read_text())
        counts.update(data["counts"])
        overlap.update(data["overlap"])
        reasons.update(data["reasons"])
        for bucket in samples:
            samples[bucket].extend(data["samples"][bucket])
    return counts, overlap, reasons, samples


def iter_accepted(work_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((work_dir / "accepted").glob("part-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_source_artifacts(cache_dir: str | None, revision: str) -> dict[str, Any]:
    if not cache_dir:
        return {"status": "cache_dir_unset", "files": []}
    snapshot = Path(cache_dir) / "hub" / "datasets--open-thoughts--OpenThoughts3-1.2M" / "snapshots" / revision
    if not snapshot.is_dir():
        return {"status": "snapshot_path_not_found", "snapshot": str(snapshot), "files": []}
    files = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        resolved = path.resolve()
        files.append(
            {
                "path": str(path.relative_to(snapshot)),
                "bytes": resolved.stat().st_size,
                "content_addressed_blob": resolved.name if re.fullmatch(r"[0-9a-f]{64}", resolved.name) else None,
            }
        )
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "complete",
        "snapshot": str(snapshot),
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "total_bytes": sum(item["bytes"] for item in files),
    }


def prompt_with_instruction(prompt: str) -> str:
    clean = prompt.rstrip()
    if PROMPT_INSTRUCTION in clean:
        return clean
    return clean + "\n\n" + PROMPT_INSTRUCTION


def selection_rows(args: argparse.Namespace, work_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer

    accepted_index = {int(row["source_row_id"]): row for row in iter_accepted(work_dir)}
    source_ids = list(accepted_index)
    random.Random(args.seed).shuffle(source_ids)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    sft: list[dict[str, Any]] = []
    opd: list[dict[str, Any]] = []
    overlength = collections.Counter()
    for source_row_id in source_ids:
        row = accepted_index[source_row_id]
        if len(sft) < args.sft_size:
            rendered = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=True
            )
            if len(rendered) > 32768:
                overlength["sft_native_context"] += 1
                continue
            sft.append(
                {
                    "messages": row["messages"],
                    "metadata": {
                        "source_row_id": source_row_id,
                        "seeded_raw_row_offset": len(sft),
                        "split_seed": args.seed,
                        "rendered_tokens": len(rendered),
                    },
                }
            )
            continue
        if len(opd) < args.opd_size:
            prompt, _ = prompt_and_response(row["messages"])
            opd_prompt = prompt_with_instruction(prompt)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": opd_prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(rendered) + 64 >= 16380:
                overlength["opd_rollout_context"] += 1
                continue
            opd.append(
                {
                    "prompt": opd_prompt,
                    "metadata": {
                        "source_row_id": source_row_id,
                        "seeded_raw_row_offset": args.sft_size + len(opd),
                        "opd_reserve_offset": len(opd),
                        "split_seed": args.seed,
                        "rendered_prompt_tokens": len(rendered),
                        "prompt_instruction": PROMPT_INSTRUCTION,
                        "prompt_sha256": row["prompt_sha256"],
                        "reference_response_sha256": row["response_sha256"],
                        "source": row.get("source"),
                        "domain": row.get("domain"),
                        "difficulty": row.get("difficulty"),
                    },
                }
            )
        if len(sft) == args.sft_size and len(opd) == args.opd_size:
            break
    if len(sft) != args.sft_size or len(opd) != args.opd_size:
        raise RuntimeError(f"Unable to select requested rows: sft={len(sft)} opd={len(opd)} overlength={dict(overlength)}")
    sft_ids = {int(row["metadata"]["source_row_id"]) for row in sft}
    opd_ids = {int(row["metadata"]["source_row_id"]) for row in opd}
    if sft_ids & opd_ids:
        raise RuntimeError("SFT and OPD source-row IDs overlap")
    return sft, opd, {
        "method": "python_random_seeded_shuffle_of_accepted_raw_source_row_ids",
        "seed": args.seed,
        "overlength_exclusions": dict(overlength),
        "sft_source_ids_sha256": hashlib.sha256(json.dumps(sorted(sft_ids), separators=(",", ":")).encode()).hexdigest(),
        "opd_source_ids_sha256": hashlib.sha256(json.dumps(sorted(opd_ids), separators=(",", ":")).encode()).hexdigest(),
        "source_row_id_intersection": 0,
        "prompt_grouping": False,
        "prompt_deduplication": False,
        "repeated_prompt_text_may_cross_splits": True,
        "math500_cross_contamination_check": False,
    }


def markdown_audit(audit: dict[str, Any]) -> str:
    counts = audit["counts"]
    gate = audit["validation_gate"]
    return f"""# OpenThoughts3 cleanup audit

- Dataset: `{audit['dataset']}@{audit['revision']}`
- Raw rows: {counts['source_rows']:,}
- Incomplete think: {counts['incomplete_think']:,} (gate {gate['incomplete_think']['range']})
- Assistant responses containing CJK: {counts['assistant_contains_cjk']:,} (legacy non-English gate {gate['assistant_contains_cjk']['range']})
- Retained: {counts['retained']:,} (gate {gate['retained']['range']})
- Gate passed: `{gate['passed']}`
- Filter: any Han, Hiragana, Katakana, Hangul, or Bopomofo Unicode letter in the assistant response only
- Unicode database: {audit['filter']['unicode_version']}
- Split: seeded raw-row shuffle; 200,000 SFT then 100,000 OPD; source-row-ID overlap only
- Prompt grouping/deduplication: disabled
- MATH-500 cross-contamination check: disabled

See the JSON audit and compressed per-row classifications for full overlap and CJK-match details.
"""


def build_validation_gate(counts: collections.Counter[str], full_run: bool, minimum_eligible_rows: int) -> dict[str, Any]:
    gate: dict[str, Any] = {
        name: {
            "value": counts[name],
            "reference": REFERENCE[name],
            "range": list(RANGES[name]),
            "passed": RANGES[name][0] <= counts[name] <= RANGES[name][1],
        }
        for name in REFERENCE
    }
    gate["full_revision"] = full_run
    gate["minimum_eligible_rows"] = {
        "value": counts["retained"],
        "required": minimum_eligible_rows,
        "passed": counts["retained"] >= minimum_eligible_rows,
    }
    gate["passed"] = (
        full_run
        and gate["minimum_eligible_rows"]["passed"]
        and all(gate[name]["passed"] for name in REFERENCE)
    )
    return gate


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, revision=args.revision, split=args.split, cache_dir=args.cache_dir)
    source_rows = len(dataset)
    if args.max_source_rows is not None:
        source_rows = min(source_rows, args.max_source_rows)
    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        for start in range(0, source_rows, args.shard_size):
            process_shard(dataset, start, min(source_rows, start + args.shard_size), work_dir, pool, args.audit_samples)
    counts, overlap, reasons, samples = aggregate(work_dir)
    full_run = args.max_source_rows is None and counts["source_rows"] == len(dataset) == 1_200_000
    validation_gate = build_validation_gate(counts, full_run, args.sft_size + args.opd_size)
    audit: dict[str, Any] = {
        "schema_version": 2,
        "cleanup_version": "ot3_assistant_cjk_v1",
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "counts": dict(counts),
        "overlap_matrix": dict(overlap),
        "reason_counts": dict(reasons.most_common()),
        "validation_gate": validation_gate,
        "filter": {
            "type": "unicode_cjk_letter_presence",
            "scope": "assistant_response_only",
            "scripts": ["Han", "Hiragana", "Katakana", "Hangul", "Bopomofo"],
            "unicode_version": unicodedata.unidata_version,
            "match_policy": "any Unicode general-category Letter in the configured CJK ranges; punctuation and symbols excluded",
            "legacy_reference_mapping": "assistant_contains_cjk reuses the issue #1493 non-English/mixed reference count and +/-5% range",
        },
        "audit_samples": {key: value[: args.audit_samples] for key, value in samples.items()},
        "per_row_classifications": str(work_dir / "classifications"),
        "source_artifacts": cached_source_artifacts(args.cache_dir, args.revision),
    }
    atomic_text(Path(args.audit_json), json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(Path(args.audit_md), markdown_audit(audit))
    if not validation_gate["passed"]:
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "revision": args.revision,
                    "raw_rows": len(dataset),
                    "counts": counts,
                    "overlap": overlap,
                    "gate": validation_gate,
                    "likely_differences_to_review": [
                        "dataset field schema",
                        "dataset revision",
                        "think-tag rule",
                        "assistant-only CJK heuristic versus the legacy non-English reference semantics",
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit("OpenThoughts3 cleanup validation gate failed closed; SFT/OPD selection was not created")
    sft, opd, split_info = selection_rows(args, work_dir)
    atomic_jsonl(Path(args.sft_out), sft)
    atomic_jsonl(Path(args.opd_reserve_out), opd)
    atomic_jsonl(Path(args.opd_headline_out), opd[: args.headline_size])
    split_info.update(
        {
            "sft_rows": len(sft),
            "opd_reserve_rows": len(opd),
            "opd_headline_rows": args.headline_size,
            "outputs": {
                "sft": {"path": args.sft_out, "sha256": sha256_file(Path(args.sft_out))},
                "opd_reserve": {"path": args.opd_reserve_out, "sha256": sha256_file(Path(args.opd_reserve_out))},
                "opd_headline": {"path": args.opd_headline_out, "sha256": sha256_file(Path(args.opd_headline_out))},
            },
        }
    )
    atomic_text(Path(args.split_metadata), json.dumps({"cleanup_audit": audit, "selection": split_info}, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"validation_gate": validation_gate, "selection": split_info}, indent=2), flush=True)


if __name__ == "__main__":
    main()
