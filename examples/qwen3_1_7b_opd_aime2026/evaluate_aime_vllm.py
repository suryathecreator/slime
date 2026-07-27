#!/usr/bin/env python3
"""Four-shard, 16-sample direct-vLLM evaluation for AIME 2026."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from aime_answer_v2 import SCORER_VERSION, metrics_from_predictions, score_answer
from prompt_contract import PROMPT_CONTRACT_VERSION, prompt_messages


NUM_PROBLEMS = 30
SAMPLES_PER_PROBLEM = 16
NUM_GENERATIONS = NUM_PROBLEMS * SAMPLES_PER_PROBLEM
NUM_SHARDS = 4
REQUESTS_PER_SHARD = NUM_GENERATIONS // NUM_SHARDS
BASE_SEED = 42
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def qwen_stop_metadata(tokenizer: Any) -> dict[str, Any]:
    eos = tokenizer.eos_token_id
    eos_ids = [int(value) for value in eos] if isinstance(eos, (list, tuple)) else [int(eos)] if eos is not None else []
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if im_end is None or im_end == tokenizer.unk_token_id:
        raise ValueError("Tokenizer has no valid <|im_end|> token")
    return {
        "eos_token_ids": eos_ids,
        "im_end_token_id": int(im_end),
        "endoftext_token_id": int(endoftext) if endoftext is not None else None,
        "stop_token_ids": sorted(set(eos_ids + [int(im_end)])),
    }


def rendered_prompt(tokenizer: Any, problem: str) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        prompt_messages(problem),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return text, list(tokenizer.encode(text, add_special_tokens=False))


def request_seed(problem_position: int, sample_index: int) -> int:
    if not 0 <= problem_position < NUM_PROBLEMS or not 0 <= sample_index < SAMPLES_PER_PROBLEM:
        raise ValueError("Invalid AIME request coordinate")
    return BASE_SEED + problem_position * SAMPLES_PER_PROBLEM + sample_index


def request_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != NUM_PROBLEMS:
        raise ValueError(f"Expected {NUM_PROBLEMS} AIME problems, found {len(rows)}")
    if [int(row["problem_idx"]) for row in rows] != list(range(1, NUM_PROBLEMS + 1)):
        raise ValueError("AIME problems are not in canonical problem_idx order")
    requests: list[dict[str, Any]] = []
    for problem_position, row in enumerate(rows):
        for sample_index in range(SAMPLES_PER_PROBLEM):
            flat_index = problem_position * SAMPLES_PER_PROBLEM + sample_index
            requests.append(
                {
                    "flat_index": flat_index,
                    "problem_position": problem_position,
                    "sample_index": sample_index,
                    "seed": request_seed(problem_position, sample_index),
                    "row": row,
                }
            )
    return requests


def resolved_cudagraph_config(engine: Any, enforce_eager: bool) -> dict[str, Any]:
    llm_engine = getattr(engine, "llm_engine", None)
    vllm_config = getattr(llm_engine, "vllm_config", None)
    compilation = getattr(vllm_config, "compilation_config", None)
    if compilation is None:
        if not enforce_eager:
            raise RuntimeError("CUDA graphs requested, but vLLM exposed no compilation config")
        return {"cudagraph_mode": "unavailable", "cudagraph_capture_sizes": []}
    mode = str(getattr(compilation, "cudagraph_mode", "unknown"))
    sizes = [int(value) for value in (getattr(compilation, "cudagraph_capture_sizes", None) or [])]
    if not enforce_eager and (mode.upper() == "NONE" or not sizes):
        raise RuntimeError(f"CUDA graphs requested but disabled: mode={mode}, capture_sizes={sizes}")
    return {"cudagraph_mode": mode, "cudagraph_capture_sizes": sizes}


def sampling_policy() -> dict[str, Any]:
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
        "n": 1,
        "samples_per_problem": SAMPLES_PER_PROBLEM,
        "seed_base": BASE_SEED,
        "seed_formula": "42 + problem_position_zero_based * 16 + sample_index_zero_based",
        "metric": "16-sample Monte Carlo estimate of single-sample accuracy",
    }


def shard_fingerprint(args: argparse.Namespace, data_hash: str) -> str:
    policy = {
        "schema_version": 2,
        "scorer_version": SCORER_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "data_sha256": data_hash,
        "model": str(Path(args.model).resolve()),
        "tokenizer": str(Path(args.tokenizer or args.model).resolve()),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "max_model_len": args.max_model_len,
        "max_response_len": args.max_response_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "sampling": sampling_policy(),
    }
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def run_shard(args: argparse.Namespace) -> None:
    if args.num_shards != NUM_SHARDS or not 0 <= args.shard_index < NUM_SHARDS:
        raise ValueError("Production AIME evaluation requires shard indices 0..3 of four shards")
    data_path = Path(args.data)
    rows = read_jsonl(data_path)
    all_requests = request_grid(rows)
    requests = [item for item in all_requests if item["flat_index"] % NUM_SHARDS == args.shard_index]
    if len(requests) != REQUESTS_PER_SHARD:
        raise ValueError(f"Shard {args.shard_index} has {len(requests)} requests, expected {REQUESTS_PER_SHARD}")

    output_dir = Path(args.output_dir)
    data_hash = sha256_file(data_path)
    fingerprint = shard_fingerprint(args, data_hash)
    prediction_path = output_dir / f"shard_{args.shard_index:02d}_predictions.jsonl"
    manifest_path = output_dir / f"shard_{args.shard_index:02d}_manifest.json"
    if args.resume and prediction_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete" and manifest.get("fingerprint") == fingerprint and manifest.get("prediction_sha256") == sha256_file(prediction_path):
            if args.ready_file:
                ready = Path(args.ready_file)
                ready.parent.mkdir(parents=True, exist_ok=True)
                ready.write_text("reused\n", encoding="utf-8")
            print(f"Reusing complete shard {args.shard_index}: {manifest_path}", flush=True)
            return

    tokenizer_path = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_path)
    stop_metadata = qwen_stop_metadata(tokenizer)
    prompt_cache: dict[str, tuple[str, list[int]]] = {}
    prepared: list[dict[str, Any]] = []
    for request in requests:
        row = request["row"]
        problem_id = str(row["problem_id"])
        if problem_id not in prompt_cache:
            prompt_cache[problem_id] = rendered_prompt(tokenizer, str(row["problem"]))
        prompt, prompt_ids = prompt_cache[problem_id]
        max_tokens = min(args.max_response_len, args.max_model_len - len(prompt_ids))
        if max_tokens <= 0:
            raise ValueError(f"AIME prompt {problem_id} leaves no generation capacity")
        prepared.append({**request, "rendered_prompt": prompt, "prompt_token_count": len(prompt_ids), "max_tokens": max_tokens})

    from vllm import LLM, SamplingParams

    engine_started = time.time()
    engine = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=args.enforce_eager,
        seed=BASE_SEED,
    )
    graph_config = resolved_cudagraph_config(engine, args.enforce_eager)
    engine_ready = time.time()
    if args.ready_file:
        ready = Path(args.ready_file)
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text(f"{engine_ready}\n", encoding="utf-8")
    print(f"VLLM_ENGINE_READY shard={args.shard_index} load_seconds={engine_ready - engine_started:.3f}", flush=True)

    params = [
        SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            min_p=MIN_P,
            n=1,
            seed=int(item["seed"]),
            max_tokens=int(item["max_tokens"]),
            stop_token_ids=stop_metadata["stop_token_ids"],
            ignore_eos=False,
        )
        for item in prepared
    ]
    generation_started = time.time()
    outputs = engine.generate([item["rendered_prompt"] for item in prepared], sampling_params=params, use_tqdm=True)
    generation_finished = time.time()
    if len(outputs) != len(prepared):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prepared)} requests")

    predictions: list[dict[str, Any]] = []
    for item, request_output in zip(prepared, outputs, strict=True):
        if len(request_output.outputs) != 1:
            raise RuntimeError("Each independent AIME request must return exactly one generation")
        generated = request_output.outputs[0]
        generated_text = str(generated.text)
        generated_tokens = len(generated.token_ids)
        finish_reason = str(generated.finish_reason) if generated.finish_reason is not None else None
        cap_hit = finish_reason == "length" or generated_tokens >= int(item["max_tokens"])
        row = item["row"]
        prediction = {
            "problem_id": str(row["problem_id"]),
            "problem_idx": int(row["problem_idx"]),
            "sample_index": int(item["sample_index"]),
            "flat_index": int(item["flat_index"]),
            "seed": int(item["seed"]),
            "problem": row["problem"],
            "correct_answer": int(row["correct_answer"]),
            "generated_text": generated_text,
            "generated_token_count": generated_tokens,
            "prompt_token_count": int(item["prompt_token_count"]),
            "max_tokens": int(item["max_tokens"]),
            "finish_reason": finish_reason,
            "stop_reason": json_safe(generated.stop_reason),
            "cap_hit": cap_hit,
            "model": str(Path(args.model).resolve()),
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "sampling": {**sampling_policy(), "seed": int(item["seed"])},
        }
        prediction.update(
            score_answer(
                generated_text,
                int(row["correct_answer"]),
                str(row["problem"]),
                str(row["problem_id"]),
                cap_hit,
                finish_reason,
            )
        )
        predictions.append(prediction)

    write_jsonl_atomic(prediction_path, predictions)
    manifest = {
        "status": "complete",
        "fingerprint": fingerprint,
        "schema_version": 2,
        "scorer_version": SCORER_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "data": str(data_path.resolve()),
        "data_sha256": data_hash,
        "model": str(Path(args.model).resolve()),
        "tokenizer": str(Path(tokenizer_path).resolve()),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_requests": len(prepared),
        "flat_indices": [int(item["flat_index"]) for item in prepared],
        "sampling_config": sampling_policy(),
        "engine_config": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": args.enforce_eager,
            **graph_config,
        },
        "stop_tokens": stop_metadata,
        "engine_load_seconds": engine_ready - engine_started,
        "generation_start_epoch": generation_started,
        "generation_end_epoch": generation_finished,
        "generation_seconds": generation_finished - generation_started,
        "generated_token_count": sum(int(row["generated_token_count"]) for row in predictions),
        "prediction_file": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
    }
    write_json_atomic(manifest_path, manifest)


def load_manifests(stage_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for shard_index in range(NUM_SHARDS):
        path = stage_dir / f"shard_{shard_index:02d}_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("shard_index") != shard_index:
            raise ValueError(f"Invalid shard manifest: {path}")
        prediction_path = stage_dir / f"shard_{shard_index:02d}_predictions.jsonl"
        if manifest.get("prediction_sha256") != sha256_file(prediction_path):
            raise ValueError(f"Prediction hash mismatch for shard {shard_index}")
        manifests.append(manifest)
    for key in ("data_sha256", "model", "tokenizer", "scorer_version", "prompt_contract_version", "sampling_config", "engine_config"):
        if len({json.dumps(manifest.get(key), sort_keys=True) for manifest in manifests}) != 1:
            raise ValueError(f"Shard manifests disagree on {key}")
    return manifests


def merge(args: argparse.Namespace) -> None:
    stage_dir = Path(args.stage_dir)
    data_path = Path(args.data)
    data_rows = read_jsonl(data_path)
    request_grid(data_rows)
    manifests = load_manifests(stage_dir)
    if manifests[0]["data_sha256"] != sha256_file(data_path):
        raise ValueError("Merge dataset does not match shard manifests")
    predictions: list[dict[str, Any]] = []
    for shard_index in range(NUM_SHARDS):
        predictions.extend(read_jsonl(stage_dir / f"shard_{shard_index:02d}_predictions.jsonl"))
    predictions.sort(key=lambda row: int(row["flat_index"]))
    if len(predictions) != NUM_GENERATIONS or [int(row["flat_index"]) for row in predictions] != list(range(NUM_GENERATIONS)):
        raise ValueError("AIME merge does not cover each of the 480 canonical requests exactly once")
    for row in predictions:
        expected_seed = request_seed(int(row["problem_idx"]) - 1, int(row["sample_index"]))
        if int(row["seed"]) != expected_seed:
            raise ValueError(f"Seed mismatch for {row['problem_id']} sample {row['sample_index']}")

    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    base_metrics = metrics_from_predictions(predictions)
    per_problem: list[dict[str, Any]] = []
    for data_row in data_rows:
        problem_rows = [row for row in predictions if row["problem_id"] == data_row["problem_id"]]
        if len(problem_rows) != SAMPLES_PER_PROBLEM or sorted(int(row["sample_index"]) for row in problem_rows) != list(range(SAMPLES_PER_PROBLEM)):
            raise ValueError(f"Incomplete sample grid for {data_row['problem_id']}")
        correct = sum(bool(row["is_correct"]) for row in problem_rows)
        per_problem.append(
            {
                "problem_id": data_row["problem_id"],
                "problem_idx": int(data_row["problem_idx"]),
                "correct_count": correct,
                "num_samples": SAMPLES_PER_PROBLEM,
                "pass_at_1_mc": correct / SAMPLES_PER_PROBLEM,
            }
        )
    generated = sum(int(manifest["generated_token_count"]) for manifest in manifests)
    start = min(float(manifest["generation_start_epoch"]) for manifest in manifests)
    end = max(float(manifest["generation_end_epoch"]) for manifest in manifests)
    makespan = end - start
    metrics = {
        "status": "complete",
        "benchmark": "MathArena/aime_2026",
        "num_problems": NUM_PROBLEMS,
        "samples_per_problem": SAMPLES_PER_PROBLEM,
        "num_generations": NUM_GENERATIONS,
        "metric_name": "pass_at_1_mc_16",
        "metric_semantics": "Mean per-problem fraction correct across 16 independent samples; equivalently total correct / 480. This is not pass@16 or majority-vote accuracy.",
        **base_metrics,
        "per_problem": per_problem,
        "sampling_config": sampling_policy(),
        "scorer_version": SCORER_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "data_sha256": sha256_file(data_path),
        "predictions_sha256": sha256_file(prediction_path),
        "model": manifests[0]["model"],
        "engine_config": manifests[0]["engine_config"],
        "stop_tokens": manifests[0]["stop_tokens"],
        "aggregate_generated_tokens": generated,
        "four_worker_generation_makespan_seconds": makespan,
        "aggregate_generated_tokens_per_second": generated / makespan if makespan else 0.0,
        "average_generated_length": generated / NUM_GENERATIONS,
        "engine_load_seconds_by_shard": [manifest["engine_load_seconds"] for manifest in manifests],
        "finish_reason_counts": dict(sorted(Counter(str(row.get("finish_reason")) for row in predictions).items())),
    }
    if abs(float(metrics["pass_at_1_mc"]) - sum(row["pass_at_1_mc"] for row in per_problem) / NUM_PROBLEMS) > 1e-12:
        raise AssertionError("Aggregate and mean per-problem pass@1 estimates disagree")
    write_json_atomic(output_dir / "metrics.json", metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--model", required=True)
    shard.add_argument("--tokenizer")
    shard.add_argument("--data", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    shard.add_argument("--max-model-len", type=int, default=32768)
    shard.add_argument("--max-response-len", type=int, default=31744)
    shard.add_argument("--max-num-seqs", type=int, required=True)
    shard.add_argument("--max-num-batched-tokens", type=int, default=16384)
    shard.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    shard.add_argument("--enforce-eager", action="store_true")
    shard.add_argument("--resume", action="store_true")
    shard.add_argument("--ready-file")
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--stage-dir", required=True)
    merge_parser.add_argument("--data", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "shard":
        run_shard(args)
    elif args.command == "merge":
        merge(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
