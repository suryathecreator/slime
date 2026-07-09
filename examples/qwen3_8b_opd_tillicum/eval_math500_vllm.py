#!/usr/bin/env python3
"""Run or merge a vLLM MATH-500 eval shard."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    shard = sub.add_parser("shard")
    shard.add_argument("--model", required=True)
    shard.add_argument("--data", required=True)
    shard.add_argument("--out", required=True)
    shard.add_argument("--stage", required=True)
    shard.add_argument("--train-samples", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--num-shards", type=int, required=True)
    shard.add_argument("--max-response-len", type=int, required=True)
    shard.add_argument("--max-context-len", type=int, required=True)
    shard.add_argument("--max-model-len", type=int, required=True)
    shard.add_argument("--gpu-memory-utilization", type=float, required=True)
    shard.add_argument("--max-num-seqs", type=int, required=True)
    shard.add_argument("--max-num-batched-tokens", type=int, required=True)
    shard.add_argument("--dtype", default="bfloat16")
    shard.add_argument("--trust-remote-code", action="store_true")
    shard.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    shard.add_argument("--chunk-size", type=int, default=16)
    shard.add_argument("--resume-completed", action="store_true")

    merge = sub.add_parser("merge")
    merge.add_argument("--stage-dir", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--expected-samples", type=int, required=True)

    prep = sub.add_parser("prepare-math500")
    prep.add_argument("--out", required=True)
    prep.add_argument("--dataset", default=os.environ.get("MATH500_DATASET", "HuggingFaceH4/MATH-500"))
    prep.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    return parser.parse_args()


def write_math500_jsonl(path: Path, dataset: str, hf_home: str | None) -> None:
    if path.exists():
        return
    from datasets import load_dataset

    ds = load_dataset(dataset, split="test", cache_dir=hf_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            prompt = row.get("problem") or row.get("question") or row.get("prompt")
            label = row.get("answer") or row.get("final_answer") or row.get("label")
            if prompt is None or label is None:
                raise RuntimeError(f"Cannot infer prompt/label fields from MATH-500 row keys: {list(row.keys())}")
            f.write(json.dumps({"prompt": prompt, "label": label, "metadata": {"source_row_id": idx}}, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            item["_eval_index"] = idx
            rows.append(item)
    return rows


def build_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def chunk_path(stage_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> Path:
    first = rows[0]["_eval_index"]
    last = rows[-1]["_eval_index"]
    return stage_dir / f"debug_eval_chunk_shard{shard_index}_start{first}_end{last}.pt"


def load_valid_chunk(path: Path, expected_indices: set[int]) -> list[dict[str, Any]] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Ignoring unreadable vLLM chunk {path}: {exc}")
        return None
    samples = payload.get("samples", [])
    found_indices = {int(sample["eval_index"]) for sample in samples}
    if found_indices != expected_indices or len(samples) != len(expected_indices):
        print(
            f"Ignoring incomplete vLLM chunk {path}: "
            f"expected_indices={sorted(expected_indices)} found_indices={sorted(found_indices)} samples={len(samples)}"
        )
        return None
    return samples


def save_chunk_atomic(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def prepare_chunk_rows(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[tuple[int, str, dict[str, Any]]], list[dict[str, Any]]]:
    prepared = []
    zero_cap_samples = []
    for row in rows:
        rendered_prompt = build_prompt(tokenizer, str(row["prompt"]))
        prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
        available = args.max_context_len - len(prompt_ids)
        effective_cap = min(args.max_response_len, max(available, 0))
        base = {
            "eval_index": row["_eval_index"],
            "prompt": row["prompt"],
            "label": row["label"],
            "metadata": {
                **(row.get("metadata") or {}),
                "stage": args.stage,
                "train_samples": args.train_samples,
                "prompt_tokens": len(prompt_ids),
                "requested_max_new_tokens": args.max_response_len,
                "effective_max_new_tokens": effective_cap,
                "max_context_len": args.max_context_len,
                "cap_clamped": effective_cap != args.max_response_len,
            },
        }
        if effective_cap <= 0:
            zero_cap_samples.append({**base, "response": "", "response_length": 0, "status": "truncated"})
        else:
            prepared.append((effective_cap, rendered_prompt, base))
    return prepared, zero_cap_samples


def run_chunk(
    llm: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    prepared, zero_cap_samples = prepare_chunk_rows(tokenizer, rows, args)
    samples = list(zero_cap_samples)
    grouped: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for cap, prompt, base in prepared:
        grouped[cap].append((prompt, base))

    for cap, items in sorted(grouped.items()):
        outputs = llm.generate(
            [prompt for prompt, _ in items],
            SamplingParams(temperature=0.0, top_p=1.0, max_tokens=cap),
            use_tqdm=True,
        )
        for output, (_, base) in zip(outputs, items, strict=True):
            completion = output.outputs[0]
            token_ids = list(getattr(completion, "token_ids", []) or [])
            finish_reason = str(getattr(completion, "finish_reason", ""))
            status = "truncated" if finish_reason == "length" or len(token_ids) >= cap else "completed"
            samples.append(
                {
                    **base,
                    "response": completion.text,
                    "response_length": len(token_ids),
                    "status": status,
                    "finish_reason": finish_reason,
                }
            )
    samples.sort(key=lambda item: item["eval_index"])
    return samples


def run_shard(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}")

    data_path = Path(args.data)
    rows = [row for row in load_rows(data_path) if row["_eval_index"] % args.num_shards == args.shard_index]
    stage_dir = Path(args.out).parent
    chunks = [rows[index : index + args.chunk_size] for index in range(0, len(rows), args.chunk_size)]

    pending_chunks = []
    for chunk in chunks:
        path = chunk_path(stage_dir, args.shard_index, chunk)
        expected_indices = {int(row["_eval_index"]) for row in chunk}
        if args.resume_completed and load_valid_chunk(path, expected_indices) is not None:
            print(f"VLLM_EVAL_CHUNK_SKIP shard={args.shard_index} path={path} samples={len(expected_indices)}")
            continue
        pending_chunks.append((chunk, path, expected_indices))

    if not pending_chunks:
        print(f"VLLM_EVAL_SHARD_COMPLETE shard={args.shard_index} chunks={len(chunks)}")
        return

    from transformers import AutoTokenizer
    from slime.backends.vllm_utils.native_sampler import maybe_force_native_sampler

    if maybe_force_native_sampler():
        print("VLLM_EVAL_FORCE_NATIVE_SAMPLER=1: patched vLLM V1 sampler in eval parent")
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_home, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=1,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    for chunk, path, expected_indices in pending_chunks:
        samples = run_chunk(llm, tokenizer, chunk, args)
        found_indices = {int(sample["eval_index"]) for sample in samples}
        if found_indices != expected_indices:
            raise RuntimeError(
                f"Chunk index mismatch for {path}: expected {sorted(expected_indices)}, found {sorted(found_indices)}"
            )
        payload = {
            "samples": samples,
            "stage": args.stage,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "chunk_start_eval_index": min(expected_indices),
            "chunk_end_eval_index": max(expected_indices),
        }
        save_chunk_atomic(path, payload)
        print(
            f"Wrote vLLM chunk shard={args.shard_index}/{args.num_shards} "
            f"path={path} samples={len(samples)}"
        )


def merge_shards(args: argparse.Namespace) -> None:
    import torch

    stage_dir = Path(args.stage_dir)
    by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(stage_dir.glob("debug_eval_shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            eval_index = int(sample["eval_index"])
            if eval_index in by_index:
                raise RuntimeError(
                    f"Duplicate eval_index={eval_index} in {path}; already found in {by_index[eval_index][0]}"
                )
            by_index[eval_index] = (path, sample)
    for path in sorted(stage_dir.glob("debug_eval_chunk_shard*_start*_end*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            eval_index = int(sample["eval_index"])
            if eval_index in by_index:
                raise RuntimeError(
                    f"Duplicate eval_index={eval_index} in {path}; already found in {by_index[eval_index][0]}"
                )
            by_index[eval_index] = (path, sample)

    samples = [sample for _, sample in sorted(by_index.values(), key=lambda item: int(item[1]["eval_index"]))]
    expected_indices = set(range(args.expected_samples))
    if len(samples) != args.expected_samples or set(by_index) != expected_indices:
        missing = sorted(expected_indices - set(by_index))[:20]
        extra = sorted(set(by_index) - expected_indices)[:20]
        raise RuntimeError(
            f"Expected {args.expected_samples} merged samples, found {len(samples)}; "
            f"first_missing_eval_indices={missing}; first_extra_eval_indices={extra}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples}, out)
    print(f"Wrote merged vLLM debug file: {out} samples={len(samples)}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare-math500":
        write_math500_jsonl(Path(args.out), args.dataset, args.hf_home)
    elif args.command == "shard":
        run_shard(args)
    elif args.command == "merge":
        merge_shards(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
