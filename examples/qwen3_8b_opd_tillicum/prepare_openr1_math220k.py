#!/usr/bin/env python3
"""Prepare deterministic OpenR1-Math-220k SFT and OPD splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=env.get("OPENR1_DATASET", "open-r1/OpenR1-Math-220k"))
    parser.add_argument("--config", default=env.get("OPENR1_CONFIG", "default"))
    parser.add_argument("--split", default=env.get("OPENR1_SPLIT", "train"))
    parser.add_argument(
        "--revision",
        default=env.get("OPENR1_REVISION", "dc748648036c1ed619b020e056dc4b603eb39817"),
    )
    parser.add_argument("--seed", type=int, default=int(env.get("DATA_SEED", "1234")))
    parser.add_argument("--sft-size", type=int, default=int(env.get("SFT_SIZE", "50000")))
    parser.add_argument("--opd-first-size", type=int, default=int(env.get("CLEANED_OPD_FIRST_SIZE", "1024")))
    parser.add_argument("--opd-next-size", type=int, default=int(env.get("CLEANED_OPD_NEXT_SIZE", "4096")))
    parser.add_argument("--sft-out", default=env.get("SFT_PARQUET"))
    parser.add_argument("--opd-reserve-out", default=env.get("CLEANED_OPD_RESERVE_JSONL"))
    parser.add_argument("--opd-first-out", default=env.get("CLEANED_OPD_1K_JSONL"))
    parser.add_argument("--opd-next-out", default=env.get("CLEANED_OPD_4K_JSONL"))
    parser.add_argument("--metadata-out", default=env.get("SPLIT_METADATA"))
    parser.add_argument("--success-out", default=env.get("OPENR1_SUCCESS_FILE"))
    parser.add_argument("--tokenizer", default=env.get("STUDENT_HF_DIR", "Qwen/Qwen3-8B-Base"))
    parser.add_argument("--hf-home", default=env.get("HF_HOME"))
    parser.add_argument("--im-end-token-id", type=int, default=int(env.get("QWEN3_IM_END_TOKEN_ID", "151645")))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    required = (
        "sft_out",
        "opd_reserve_out",
        "opd_first_out",
        "opd_next_out",
        "metadata_out",
        "success_out",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("missing output path(s): " + ", ".join(missing))
    return args


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def value_at(values: Any, index: int, default: Any = None) -> Any:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or index >= len(values):
        return default
    return values[index]


def has_ordered_think_tags(text: str) -> bool:
    start = text.find("<think>")
    end = text.find("</think>")
    return start >= 0 and end > start


def correct_generation_candidates(row: dict[str, Any]) -> list[tuple[int, tuple[str, ...]]]:
    generations = row.get("generations") or []
    candidates: list[tuple[int, tuple[str, ...]]] = []
    for index, generation in enumerate(generations):
        if not isinstance(generation, str) or not has_ordered_think_tags(generation):
            continue
        if not bool(value_at(row.get("is_reasoning_complete"), index, False)):
            continue
        sources: list[str] = []
        if bool(value_at(row.get("correctness_math_verify"), index, False)):
            sources.append("math_verify")
        if bool(value_at(row.get("correctness_llama"), index, False)):
            sources.append("llama_judge")
        if sources:
            candidates.append((index, tuple(sources)))
    return candidates


def extract_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", message.get("from", ""))).lower()
            if role in {"user", "human"}:
                content = message.get("content", message.get("value", ""))
                if content:
                    return str(content)
    problem = row.get("problem")
    if problem:
        return str(problem)
    raise ValueError("OpenR1 row has no usable user prompt")


def stable_trace_choice(seed: int, row_key: str, candidates: list[tuple[int, tuple[str, ...]]]) -> tuple[int, tuple[str, ...]]:
    digest = hashlib.sha256(f"{seed}:{row_key}".encode()).digest()
    choice_seed = int.from_bytes(digest[:8], "big")
    return random.Random(choice_seed).choice(candidates)


def select_records(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    sft_size: int,
    opd_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    seen_prompt_hashes: set[str] = set()
    counters = {"source_rows": len(rows), "no_correct_complete_trace": 0, "duplicate_prompt": 0}

    for source_row_id in order:
        row = dict(rows[source_row_id])
        candidates = correct_generation_candidates(row)
        if not candidates:
            counters["no_correct_complete_trace"] += 1
            continue
        prompt = extract_prompt(row)
        prompt_hash = sha256_text(prompt)
        uuid = str(row.get("uuid") or f"source_row_{source_row_id}")
        if uuid in seen_uuids or prompt_hash in seen_prompt_hashes:
            counters["duplicate_prompt"] += 1
            continue
        generation_index, correctness_sources = stable_trace_choice(seed, uuid, candidates)
        response = str(row["generations"][generation_index])
        selected.append(
            {
                "source_row_id": source_row_id,
                "uuid": uuid,
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "response": response,
                "response_sha256": sha256_text(response),
                "generation_index": generation_index,
                "correctness_sources": list(correctness_sources),
                "candidate_generation_indices": [index for index, _ in candidates],
                "source": row.get("source"),
                "problem_type": row.get("problem_type"),
                "question_type": row.get("question_type"),
                "answer": row.get("answer"),
            }
        )
        seen_uuids.add(uuid)
        seen_prompt_hashes.add(prompt_hash)
        if len(selected) == sft_size + opd_size:
            break

    if len(selected) != sft_size + opd_size:
        raise RuntimeError(f"Needed {sft_size + opd_size} eligible unique rows, found {len(selected)}")
    return selected[:sft_size], selected[sft_size:], counters


def split_metadata(record: dict[str, Any], split: str, offset: int, seed: int) -> dict[str, Any]:
    return {
        "source_dataset": "open-r1/OpenR1-Math-220k",
        "source_row_id": record["source_row_id"],
        "uuid": record["uuid"],
        "prompt_sha256": record["prompt_sha256"],
        "response_sha256": record["response_sha256"],
        "generation_index": record["generation_index"],
        "candidate_generation_indices": record["candidate_generation_indices"],
        "correctness_sources": record["correctness_sources"],
        "source": record["source"],
        "problem_type": record["problem_type"],
        "question_type": record["question_type"],
        "selection_seed": seed,
        "selected_split": split,
        "selected_offset": offset,
    }


def build_sft_rows(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": record["response"]},
            ],
            "metadata": split_metadata(record, "sft", offset, seed),
        }
        for offset, record in enumerate(records)
    ]


def build_opd_rows(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return [
        {
            "prompt": record["prompt"],
            "reference_response": record["response"],
            "reference_answer": record["answer"],
            "metadata": split_metadata(record, "opd", offset, seed),
        }
        for offset, record in enumerate(records)
    ]


def validate_chat_targets(tokenizer: Any, sft_rows: list[dict[str, Any]], im_end_token_id: int) -> list[dict[str, Any]]:
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if token_id != im_end_token_id:
        raise RuntimeError(f"Tokenizer maps <|im_end|> to {token_id}, expected {im_end_token_id}")
    audit: list[dict[str, Any]] = []
    for index in (0, len(sft_rows) // 2, len(sft_rows) - 1):
        row = sft_rows[index]
        token_ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if not token_ids or token_ids[-1] != im_end_token_id:
            raise RuntimeError(f"SFT row {index} does not render with terminal <|im_end|>")
        assistant = row["messages"][-1]["content"]
        if not has_ordered_think_tags(assistant):
            raise RuntimeError(f"SFT row {index} lost its thinking tags")
        audit.append(
            {
                "sft_offset": index,
                "source_row_id": row["metadata"]["source_row_id"],
                "rendered_tokens": len(token_ids),
                "terminal_token_id": token_ids[-1],
                "think_start": assistant.find("<think>"),
                "think_end": assistant.find("</think>"),
            }
        )
    return audit


def hash_ids(rows: list[dict[str, Any]]) -> str:
    ids = [row["metadata"]["source_row_id"] for row in rows]
    return sha256_text(json.dumps(ids, separators=(",", ":")))


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    args = parse_args()
    outputs = [
        Path(args.sft_out),
        Path(args.opd_reserve_out),
        Path(args.opd_first_out),
        Path(args.opd_next_out),
        Path(args.metadata_out),
        Path(args.success_out),
    ]
    if all(path.exists() for path in outputs) and not args.force:
        print("All OpenR1 preparation outputs already exist; nothing to do.")
        return

    print(f"Loading {args.dataset} config={args.config} split={args.split} revision={args.revision or 'default'}")
    load_kwargs: dict[str, Any] = {"split": args.split, "cache_dir": args.hf_home}
    if args.revision:
        load_kwargs["revision"] = args.revision
    dataset = load_dataset(args.dataset, args.config, **load_kwargs)
    opd_total = args.opd_first_size + args.opd_next_size
    sft_records, opd_records, counters = select_records(
        dataset,
        seed=args.seed,
        sft_size=args.sft_size,
        opd_size=opd_total,
    )
    sft_rows = build_sft_rows(sft_records, args.seed)
    opd_rows = build_opd_rows(opd_records, args.seed)
    opd_first = opd_rows[: args.opd_first_size]
    opd_next = opd_rows[args.opd_first_size :]

    sft_ids = {row["metadata"]["source_row_id"] for row in sft_rows}
    opd_ids = {row["metadata"]["source_row_id"] for row in opd_rows}
    sft_uuids = {row["metadata"]["uuid"] for row in sft_rows}
    opd_uuids = {row["metadata"]["uuid"] for row in opd_rows}
    sft_hashes = {row["metadata"]["prompt_sha256"] for row in sft_rows}
    opd_hashes = {row["metadata"]["prompt_sha256"] for row in opd_rows}
    if sft_ids & opd_ids or sft_uuids & opd_uuids or sft_hashes & opd_hashes:
        raise RuntimeError("OpenR1 SFT/OPD overlap detected")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.hf_home, trust_remote_code=True)
    chat_audit = validate_chat_targets(tokenizer, sft_rows, args.im_end_token_id)

    write_jsonl_atomic(Path(args.sft_out), sft_rows)
    write_jsonl_atomic(Path(args.opd_reserve_out), opd_rows)
    write_jsonl_atomic(Path(args.opd_first_out), opd_first)
    write_jsonl_atomic(Path(args.opd_next_out), opd_next)

    metadata = {
        "schema_version": 1,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "revision": args.revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "seed": args.seed,
        "selection": {
            "method": "seeded_prompt_shuffle_then_seeded_uniform_correct_trace_choice",
            "correct_trace_rule": "reasoning_complete AND (math_verify OR llama_judge) AND ordered think tags",
            "sft": len(sft_rows),
            "opd_first": len(opd_first),
            "opd_next": len(opd_next),
            "opd_total": len(opd_rows),
            **counters,
        },
        "disjointness": {
            "source_id_intersection": len(sft_ids & opd_ids),
            "uuid_intersection": len(sft_uuids & opd_uuids),
            "prompt_hash_intersection": len(sft_hashes & opd_hashes),
            "sft_source_ids_sha256": hash_ids(sft_rows),
            "opd_source_ids_sha256": hash_ids(opd_rows),
        },
        "chat_contract": {
            "enable_thinking": True,
            "loss_mask_type": "qwen3",
            "assistant_terminal_token": "<|im_end|>",
            "assistant_terminal_token_id": args.im_end_token_id,
            "accepted_generation_stop_token_ids": [151645, 151643],
            "audit": chat_audit,
        },
        "outputs": {
            "sft": args.sft_out,
            "opd_reserve": args.opd_reserve_out,
            "opd_first": args.opd_first_out,
            "opd_next": args.opd_next_out,
        },
    }
    atomic_text(Path(args.metadata_out), json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    atomic_text(Path(args.success_out), json.dumps({"metadata": args.metadata_out, "counts": metadata["selection"]}) + "\n")
    print(json.dumps({"selection": metadata["selection"], "disjointness": metadata["disjointness"]}, indent=2))


if __name__ == "__main__":
    main()
