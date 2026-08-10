#!/usr/bin/env python3
"""Run one resumable sampled shard for the masked Qwen2.5-7B evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


TARGET_CONTEXT = 32768
STOP_TOKENS = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
PROMPT_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rendered_messages(problem: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"{PROMPT_INSTRUCTION}\n\nProblem:\n{problem}",
        }
    ]


def effective_response_budget(rendered_prompt_tokens: int) -> int:
    budget = TARGET_CONTEXT - rendered_prompt_tokens
    if rendered_prompt_tokens < 1 or budget < 1:
        raise RuntimeError(
            f"prompt leaves no response budget: {TARGET_CONTEXT} - "
            f"{rendered_prompt_tokens} = {budget}"
        )
    return budget


def sample_seed(repeat: int, eval_index: int) -> int:
    if repeat not in range(3):
        raise ValueError(f"invalid sampled repeat: {repeat}")
    return 1234 + repeat * 500 + eval_index


def decoding() -> dict[str, float | int]:
    return {"temperature": 0.7, "top_p": 0.8, "top_k": -1}


def policy(
    args: argparse.Namespace, checkpoint_identity: str, eval_file_sha256: str
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "checkpoint_manifest_sha256": checkpoint_identity,
        "decoding": decoding(),
        "eval_file_sha256": eval_file_sha256,
        "mode": "sampled",
        "prompt": {
            "add_generation_prompt": True,
            "instruction": PROMPT_INSTRUCTION,
            "layout": "instruction + two newlines + Problem: + problem",
        },
        "repeat": args.repeat,
        "response_budget": "32768 - rendered_prompt_tokens",
        "seed_formula": "1234 + repeat * 500 + eval_index",
        "stop_token_ids": sorted(STOP_TOKENS.values()),
        "target": args.target,
        "target_context": TARGET_CONTEXT,
        "tokenizer_family": args.tokenizer_family,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconcile_records(
    path: Path,
    expected_indices: list[int],
    expected_policy_sha256: str,
) -> dict[int, dict[str, Any]]:
    expected = set(expected_indices)
    records: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        index = int(row.get("eval_index", -1))
        if index in expected and row.get("policy_sha256") == expected_policy_sha256:
            records[index] = row
    ordered = [records[index] for index in expected_indices if index in records]
    if path.exists() or ordered:
        atomic_text(
            path,
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in ordered
            ),
        )
    return records


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    eval_file = Path(args.eval_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_jsonl(eval_file)
    if not rows or len(rows) > 125:
        raise RuntimeError(f"eval shard must contain 1..125 rows, found {len(rows)}")
    indices = [int(row["eval_index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise RuntimeError("eval shard contains duplicate eval indices")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, use_fast=True
    )
    actual_stops = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in STOP_TOKENS
    }
    if actual_stops != STOP_TOKENS or int(tokenizer.eos_token_id) != 151643:
        raise RuntimeError(
            f"tokenizer stop-token contract changed: actual={actual_stops} "
            f"eos={tokenizer.eos_token_id}"
        )

    generation_policy = policy(
        args, args.checkpoint_manifest_sha256, sha256_file(eval_file)
    )
    fingerprint = hashlib.sha256(
        json.dumps(generation_policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policy_path = output_dir / "generation_policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != generation_policy:
        raise RuntimeError(f"refusing incompatible resume policy: {policy_path}")
    atomic_text(policy_path, canonical_json(generation_policy))
    records_path = output_dir / "records.jsonl"
    completed = reconcile_records(records_path, indices, fingerprint)
    pending = [row for row in rows if int(row["eval_index"]) not in completed]
    if not pending:
        print(f"MATH500_GENERATION_REUSED target={args.target} mode={args.mode} repeat={args.repeat}")
        return

    prompt_records: list[tuple[dict[str, Any], list[int], str, int]] = []
    for row in pending:
        rendered = tokenizer.apply_chat_template(
            rendered_messages(str(row["problem"])),
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompt_records.append(
            (row, [int(value) for value in input_ids], rendered, effective_response_budget(len(input_ids)))
        )

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=TARGET_CONTEXT,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=0,
        enable_prefix_caching=True,
    )
    params = decoding()
    try:
        for start in range(0, len(prompt_records), args.chunk_size):
            chunk = prompt_records[start : start + args.chunk_size]
            prompts = [TokensPrompt(prompt_token_ids=item[1]) for item in chunk]
            sampling = [
                SamplingParams(
                    n=1,
                    temperature=float(params["temperature"]),
                    top_p=float(params["top_p"]),
                    top_k=int(params["top_k"]),
                    max_tokens=item[3],
                    seed=sample_seed(args.repeat, int(item[0]["eval_index"])),
                    stop_token_ids=sorted(STOP_TOKENS.values()),
                    ignore_eos=False,
                )
                for item in chunk
            ]
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
            if len(outputs) != len(chunk):
                raise RuntimeError("vLLM returned a different number of outputs")
            for (row, input_ids, rendered, budget), output in zip(chunk, outputs, strict=True):
                completion = output.outputs[0]
                token_ids = [int(value) for value in (completion.token_ids or [])]
                finish_reason = str(completion.finish_reason or "")
                stop_reason = completion.stop_reason
                if isinstance(stop_reason, int):
                    stop_reason = int(stop_reason)
                elif stop_reason is not None:
                    stop_reason = str(stop_reason)
                index = int(row["eval_index"])
                response = str(completion.text)
                record = {
                    "artifact_schema_version": 1,
                    "cap_hit": finish_reason == "length" or len(token_ids) >= budget,
                    "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
                    "effective_max_response_tokens": budget,
                    "eval_index": index,
                    "finish_reason": finish_reason,
                    "generated_token_count": len(token_ids),
                    "generated_token_ids": token_ids,
                    "mode": "sampled",
                    "policy_sha256": fingerprint,
                    "problem_id": str(row["problem_id"]),
                    "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "rendered_prompt_tokens": len(input_ids),
                    "repeat": args.repeat,
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "seed": sample_seed(args.repeat, index),
                    "stop_reason": stop_reason,
                    "target": args.target,
                    "total_context_limit": TARGET_CONTEXT,
                }
                append_jsonl(records_path, record)
                completed[index] = record
            print(
                f"MATH500_GENERATION_PROGRESS target={args.target} mode={args.mode} "
                f"repeat={args.repeat} completed={len(completed)}/{len(rows)}",
                flush=True,
            )
    finally:
        del llm

    reconcile_records(records_path, indices, fingerprint)
    if set(completed) != set(indices):
        raise RuntimeError("generation ended without every shard row")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-family", required=True, choices=["qwen2_5"])
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mode", default="sampled", choices=["sampled"])
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    sample_seed(args.repeat, 0)
    return args


if __name__ == "__main__":
    run(parse_args())
