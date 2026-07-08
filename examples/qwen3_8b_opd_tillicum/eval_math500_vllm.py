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


def run_shard(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer
    from slime.backends.vllm_utils.native_sampler import maybe_force_native_sampler

    if maybe_force_native_sampler():
        print("VLLM_EVAL_FORCE_NATIVE_SAMPLER=1: patched vLLM V1 sampler in eval parent")
    from vllm import LLM, SamplingParams

    data_path = Path(args.data)
    rows = [row for row in load_rows(data_path) if row["_eval_index"] % args.num_shards == args.shard_index]
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_home, trust_remote_code=True)

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

    samples = zero_cap_samples
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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples}, out)
    print(f"Wrote vLLM shard {args.shard_index}/{args.num_shards}: {out} samples={len(samples)}")


def merge_shards(args: argparse.Namespace) -> None:
    import torch

    stage_dir = Path(args.stage_dir)
    samples = []
    for path in sorted(stage_dir.glob("debug_eval_shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples.extend(payload.get("samples", []))
    samples.sort(key=lambda item: item["eval_index"])
    if len(samples) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} merged samples, found {len(samples)}")
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
