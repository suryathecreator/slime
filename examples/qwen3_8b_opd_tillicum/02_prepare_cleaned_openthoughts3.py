#!/usr/bin/env python3
"""Prepare cleaned OpenThoughts3 SFT/OPD splits for the Tillicum Qwen3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import unicodedata
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "prompt": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=env.get("OT3_DATASET", "open-thoughts/OpenThoughts3-1.2M"))
    parser.add_argument("--split", default=env.get("OT3_SPLIT", "train"))
    parser.add_argument("--seed", type=int, default=int(env.get("DATA_SEED", "1234")))
    parser.add_argument("--sft-size", type=int, default=int(env.get("SFT_SIZE", "25000")))
    parser.add_argument("--opd-first-size", type=int, default=int(env.get("CLEANED_OPD_FIRST_SIZE", "1024")))
    parser.add_argument("--opd-next-size", type=int, default=int(env.get("CLEANED_OPD_NEXT_SIZE", "4096")))
    parser.add_argument("--sft-out", default=env.get("SFT_PARQUET"))
    parser.add_argument("--opd-reserve-out", default=env.get("CLEANED_OPD_RESERVE_JSONL"))
    parser.add_argument("--opd-1k-out", default=env.get("CLEANED_OPD_1K_JSONL"))
    parser.add_argument("--opd-4k-out", default=env.get("CLEANED_OPD_4K_JSONL"))
    parser.add_argument("--metadata-out", default=env.get("SPLIT_METADATA"))
    parser.add_argument("--cleaned-metadata-out", default=env.get("CLEANED_METADATA"))
    parser.add_argument("--hf-home", default=env.get("HF_HOME"))
    parser.add_argument("--tokenizer", default=env.get("STUDENT_HF_REPO", "Qwen/Qwen3-8B-Base"))
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    missing = [
        name
        for name in ("sft_out", "opd_reserve_out", "opd_1k_out", "opd_4k_out", "metadata_out")
        if getattr(args, name) in (None, "")
    ]
    if missing:
        parser.error(f"missing output path(s): {', '.join(missing)}")
    return args


def first_present(row: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalize_messages(value: Any, row: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [{"role": "user", "content": value}]
    if isinstance(value, dict) and "messages" in value:
        value = value["messages"]
    if isinstance(value, list):
        out: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            raw_role = item.get("role", item.get("from", item.get("speaker", "")))
            role = ROLE_MAP.get(str(raw_role).lower(), str(raw_role).lower())
            content = item.get("content", item.get("value", item.get("text", "")))
            if role in {"system", "user", "assistant", "tool"}:
                out.append({"role": role, "content": "" if content is None else str(content)})
        if out:
            return out
    prompt = first_present(row, ["prompt", "problem", "question", "instruction", "input"])
    response = first_present(row, ["response", "completion", "answer", "solution", "output"])
    if prompt is not None and response is not None:
        return [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}]
    if prompt is not None:
        return [{"role": "user", "content": str(prompt)}]
    return []


def extract_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("messages", "conversations", "conversation"):
        if key in row and row[key] not in (None, ""):
            return normalize_messages(row[key], row)
    return normalize_messages(None, row)


def extract_prompt(row: dict[str, Any], messages: list[dict[str, str]]) -> str:
    prompt = first_present(row, ["prompt", "problem", "question", "instruction", "input"])
    if prompt is not None:
        return str(prompt)
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    raise ValueError("Could not infer prompt.")


def assistant_text(messages: list[dict[str, str]]) -> str:
    parts = [m["content"] for m in messages if m["role"] == "assistant"]
    return "\n".join(parts)


def has_complete_think_trace(text: str) -> bool:
    start = text.find("<think>")
    end = text.find("</think>")
    return start >= 0 and end > start


def script_bucket(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    if "LATIN" in name:
        return "latin"
    for marker, bucket in (
        ("CJK", "cjk"),
        ("HIRAGANA", "cjk"),
        ("KATAKANA", "cjk"),
        ("HANGUL", "cjk"),
        ("CYRILLIC", "cyrillic"),
        ("ARABIC", "arabic"),
        ("DEVANAGARI", "devanagari"),
        ("HEBREW", "hebrew"),
        ("THAI", "thai"),
    ):
        if marker in name:
            return bucket
    return "other"


def is_mixed_language(text: str) -> bool:
    counts: dict[str, int] = {}
    for ch in text:
        bucket = script_bucket(ch)
        if bucket is not None:
            counts[bucket] = counts.get(bucket, 0) + 1
    letters = sum(counts.values())
    if letters == 0:
        return False
    latin = counts.get("latin", 0)
    non_latin = letters - latin
    non_latin_scripts = [name for name, count in counts.items() if name != "latin" and count >= 10]
    if latin >= 40 and non_latin >= 20 and non_latin / letters >= 0.02:
        return True
    return len(non_latin_scripts) >= 2


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_rows(ds, label: str):
    for idx, row in enumerate(ds):
        if idx and idx % 50000 == 0:
            print(f"{label}: {idx}", flush=True)
        yield idx, row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    encoded = tokenizer(texts, add_special_tokens=False)
    return [len(ids) for ids in encoded["input_ids"]]


def token_stats(tokenizer: Any, sft_rows: list[dict[str, Any]], opd_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: list[int]) -> dict[str, float | int]:
        return {
            "count": len(values),
            "avg": float(statistics.fmean(values)) if values else 0.0,
            "max": max(values) if values else 0,
        }

    sft_texts = []
    for row in sft_rows:
        try:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])
        sft_texts.append(text)
    opd_texts = []
    for row in opd_rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = row["prompt"]
        opd_texts.append(text)

    sft_lengths: list[int] = []
    opd_lengths: list[int] = []
    for start in range(0, len(sft_texts), 256):
        sft_lengths.extend(encode_lengths(tokenizer, sft_texts[start : start + 256]))
    for start in range(0, len(opd_texts), 256):
        opd_lengths.extend(encode_lengths(tokenizer, opd_texts[start : start + 256]))
    return {"sft_rendered_token_lengths": stats(sft_lengths), "opd_prompt_token_lengths": stats(opd_lengths)}


def main() -> None:
    args = parse_args()
    output_paths = [
        Path(args.sft_out),
        Path(args.opd_reserve_out),
        Path(args.opd_1k_out),
        Path(args.opd_4k_out),
        Path(args.metadata_out),
    ]
    if args.cleaned_metadata_out:
        output_paths.append(Path(args.cleaned_metadata_out))
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        raise SystemExit("Output already exists; use --force:\n" + "\n".join(f"  {p}" for p in existing))

    print(f"Loading {args.dataset} split={args.split}", flush=True)
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.hf_home)
    if args.max_source_rows is not None:
        ds = ds.select(range(min(args.max_source_rows, len(ds))))

    counts = {
        "source_rows": len(ds),
        "incomplete_think_trace": 0,
        "mixed_language": 0,
        "valid": 0,
    }
    valid_source_ids: list[int] = []

    for idx, row in iter_rows(ds, "clean scan"):
        messages = extract_messages(row)
        text = "\n".join(m["content"] for m in messages)
        assistant = assistant_text(messages)
        incomplete = not has_complete_think_trace(assistant)
        mixed = is_mixed_language(text)
        counts["incomplete_think_trace"] += int(incomplete)
        counts["mixed_language"] += int(mixed)
        if not incomplete and not mixed:
            valid_source_ids.append(idx)
            counts["valid"] += 1

    if len(valid_source_ids) < args.sft_size + args.opd_first_size + args.opd_next_size:
        raise SystemExit(
            f"Need at least {args.sft_size + args.opd_first_size + args.opd_next_size} valid rows, "
            f"found {len(valid_source_ids)}"
        )

    rng = random.Random(args.seed)
    shuffled_valid = valid_source_ids[:]
    rng.shuffle(shuffled_valid)
    sft_reserve_count = (len(shuffled_valid) * 2) // 3
    sft_reserve_ids = shuffled_valid[:sft_reserve_count]
    opd_reserve_ids = shuffled_valid[sft_reserve_count:]
    sft_selected_ids = sft_reserve_ids[: args.sft_size]
    opd_first_ids = opd_reserve_ids[: args.opd_first_size]
    opd_next_ids = opd_reserve_ids[args.opd_first_size : args.opd_first_size + args.opd_next_size]
    opd_used_ids = opd_first_ids + opd_next_ids

    if set(sft_selected_ids) & set(opd_used_ids):
        raise RuntimeError("Selected SFT/OPD overlap detected.")

    order: dict[int, tuple[str, int]] = {}
    for i, source_id in enumerate(sft_selected_ids):
        order[source_id] = ("sft", i)
    for i, source_id in enumerate(opd_reserve_ids):
        order[source_id] = ("opd_reserve", i)

    sft_rows: list[dict[str, Any] | None] = [None] * len(sft_selected_ids)
    opd_reserve_rows: list[dict[str, Any] | None] = [None] * len(opd_reserve_ids)

    for idx, row in iter_rows(ds, "materialize"):
        item = order.get(idx)
        if item is None:
            continue
        split, pos = item
        messages = extract_messages(row)
        prompt = extract_prompt(row, messages)
        assistant = assistant_text(messages)
        metadata = {
            "source_dataset": args.dataset,
            "source_split": args.split,
            "source_row_id": idx,
            "prompt_sha256": sha256_text(prompt),
            "assistant_sha256": sha256_text(assistant),
            "cleaned_split": split,
            "cleaned_seed": args.seed,
            "cleaned_offset": pos,
        }
        if split == "sft":
            sft_rows[pos] = {"messages": messages, "metadata": metadata}
        else:
            opd_reserve_rows[pos] = {"prompt": prompt, "metadata": metadata}

    if any(row is None for row in sft_rows) or any(row is None for row in opd_reserve_rows):
        raise RuntimeError("Failed to materialize all selected rows.")

    sft_rows_typed = [row for row in sft_rows if row is not None]
    opd_rows_typed = [row for row in opd_reserve_rows if row is not None]
    opd_first_rows = opd_rows_typed[: args.opd_first_size]
    opd_next_rows = opd_rows_typed[args.opd_first_size : args.opd_first_size + args.opd_next_size]

    write_jsonl(Path(args.sft_out), sft_rows_typed)
    write_jsonl(Path(args.opd_reserve_out), opd_rows_typed)
    write_jsonl(Path(args.opd_1k_out), opd_first_rows)
    write_jsonl(Path(args.opd_4k_out), opd_next_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.hf_home, trust_remote_code=True)
    lengths = token_stats(tokenizer, sft_rows_typed, opd_first_rows + opd_next_rows)

    metadata = {
        "dataset": args.dataset,
        "source_split": args.split,
        "seed": args.seed,
        "cleaning": {
            "complete_think_required": True,
            "mixed_language_heuristic": (
                "Remove rows with substantial Latin plus non-Latin script mixture, or two substantial non-Latin scripts."
            ),
        },
        "counts": {
            **counts,
            "sft_reserve": len(sft_reserve_ids),
            "opd_reserve": len(opd_reserve_ids),
            "sft_selected": len(sft_rows_typed),
            "opd_1k_selected": len(opd_first_rows),
            "opd_next_4k_selected": len(opd_next_rows),
        },
        "sampling": {
            "method": "shuffle_valid_rows_then_2_3_1_3_split_then_take_prefixes",
            "sft_selection": "first 25000 rows from shuffled SFT reserve",
            "opd_1k_selection": "first 1024 rows from shuffled OPD reserve",
            "opd_5k_continuation_selection": "next 4096 rows from shuffled OPD reserve",
        },
        "source_row_ids": {
            "sft_selected": sft_selected_ids,
            "opd_1k": opd_first_ids,
            "opd_next_4k": opd_next_ids,
        },
        "reserve_source_row_id_sha256": {
            "sft_reserve": sha256_text(json.dumps(sft_reserve_ids, separators=(",", ":"))),
            "opd_reserve": sha256_text(json.dumps(opd_reserve_ids, separators=(",", ":"))),
        },
        "no_overlap": {
            "sft_selected_intersect_opd_used": 0,
        },
        "outputs": {
            "sft_data": args.sft_out,
            "opd_reserve_jsonl": args.opd_reserve_out,
            "opd_1k_jsonl": args.opd_1k_out,
            "opd_next_4k_jsonl": args.opd_4k_out,
        },
        "tokenizer": args.tokenizer,
        "token_lengths": lengths,
    }

    for path in {Path(args.metadata_out), Path(args.cleaned_metadata_out) if args.cleaned_metadata_out else Path(args.metadata_out)}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
