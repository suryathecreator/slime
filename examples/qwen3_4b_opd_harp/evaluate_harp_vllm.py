#!/usr/bin/env python3
"""Four-shard direct offline-vLLM HARP evaluation and authoritative merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from harp_answer_v2 import SCORER_VERSION, metrics_from_predictions, score_answer
from prompt_contract import PROMPT_CONTRACT_VERSION, prompt_messages


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
    stop_ids = sorted(set(eos_ids + [int(im_end)]))
    if int(im_end) not in stop_ids:
        raise ValueError("<|im_end|> is not in the stop-token list")
    return {
        "eos_token_ids": eos_ids,
        "im_end_token_id": int(im_end),
        "endoftext_token_id": int(endoftext) if endoftext is not None else None,
        "stop_token_ids": stop_ids,
    }


def rendered_prompt(tokenizer: Any, problem: str) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        prompt_messages(problem),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return text, list(token_ids)


def shard_fingerprint(args: argparse.Namespace, data_sha256: str, model: str) -> str:
    policy = {
        "schema_version": 1,
        "scorer_version": SCORER_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "data_sha256": data_sha256,
        "model": str(Path(model).resolve()),
        "tokenizer": str(Path(args.tokenizer or model).resolve()),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "max_model_len": args.max_model_len,
        "max_response_len": args.max_response_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "benchmark": args.benchmark,
        "benchmark_total_size": args.benchmark_total_size,
        "benchmark_max_tokens": args.benchmark_max_tokens,
    }
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode("utf-8")).hexdigest()


def run_shard(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    rows = read_jsonl(data_path)
    if len(rows) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} HARP rows, found {len(rows)}")
    if args.num_shards != 4 or args.expected_samples % args.num_shards != 0:
        raise ValueError("Production HARP evaluation requires four equal shards")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.benchmark:
        per_shard = args.benchmark_total_size // args.num_shards
        indices = indices[:per_shard]
    elif len(indices) != 125:
        raise ValueError(f"Shard {args.shard_index} has {len(indices)} rows instead of 125")

    data_hash = sha256_file(data_path)
    fingerprint = shard_fingerprint(args, data_hash, args.model)
    suffix = "benchmark" if args.benchmark else "predictions"
    prediction_path = output_dir / f"shard_{args.shard_index:02d}_{suffix}.jsonl"
    manifest_path = output_dir / f"shard_{args.shard_index:02d}_manifest.json"
    if args.resume and manifest_path.is_file() and prediction_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete" and manifest.get("fingerprint") == fingerprint:
            if args.ready_file:
                ready_path = Path(args.ready_file)
                ready_path.parent.mkdir(parents=True, exist_ok=True)
                ready_path.write_text("reused\n", encoding="utf-8")
            print(f"Reusing complete shard {args.shard_index}: {manifest_path}", flush=True)
            return

    tokenizer_path = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_path)
    stop_metadata = qwen_stop_metadata(tokenizer)
    prepared: list[dict[str, Any]] = []
    for eval_index in indices:
        row = rows[eval_index]
        prompt, prompt_ids = rendered_prompt(tokenizer, str(row["problem"]))
        available = args.max_model_len - len(prompt_ids)
        cap = min(args.max_response_len, args.benchmark_max_tokens if args.benchmark else args.max_response_len, available)
        if cap <= 0:
            raise ValueError(f"HARP prompt {row['problem_id']} leaves no generation capacity")
        prepared.append(
            {
                "eval_index": eval_index,
                "row": row,
                "rendered_prompt": prompt,
                "prompt_token_count": len(prompt_ids),
                "max_tokens": cap,
            }
        )

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
        seed=0,
    )
    engine_ready = time.time()
    if args.ready_file:
        ready_path = Path(args.ready_file)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text(f"{engine_ready}\n", encoding="utf-8")
    print(f"VLLM_ENGINE_READY shard={args.shard_index} load_seconds={engine_ready - engine_started:.3f}", flush=True)
    sampling = [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=item["max_tokens"],
            stop_token_ids=stop_metadata["stop_token_ids"],
            ignore_eos=False,
        )
        for item in prepared
    ]
    generation_started = time.time()
    outputs = engine.generate([item["rendered_prompt"] for item in prepared], sampling_params=sampling, use_tqdm=True)
    generation_finished = time.time()
    if len(outputs) != len(prepared):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prepared)} prompts")

    predictions: list[dict[str, Any]] = []
    for item, request_output in zip(prepared, outputs, strict=True):
        if len(request_output.outputs) != 1:
            raise RuntimeError(f"Expected one greedy completion, got {len(request_output.outputs)}")
        generated = request_output.outputs[0]
        generated_text = str(generated.text)
        generated_token_count = len(generated.token_ids)
        finish_reason = str(generated.finish_reason) if generated.finish_reason is not None else None
        cap_hit = finish_reason == "length" or generated_token_count >= item["max_tokens"]
        row = item["row"]
        prediction: dict[str, Any] = {
            "problem_id": str(row["problem_id"]),
            "eval_index": item["eval_index"],
            "problem": row["problem"],
            "correct_answer": row["correct_answer"],
            "generated_text": generated_text,
            "generated_token_count": generated_token_count,
            "prompt_token_count": item["prompt_token_count"],
            "max_tokens": item["max_tokens"],
            "finish_reason": finish_reason,
            "stop_reason": json_safe_stop_reason(generated.stop_reason),
            "cap_hit": cap_hit,
            "model": str(Path(args.model).resolve()),
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        }
        if not args.benchmark:
            prediction.update(score_answer(generated_text, str(row["correct_answer"])))
        predictions.append(prediction)

    write_jsonl_atomic(prediction_path, predictions)
    manifest = {
        "status": "complete",
        "fingerprint": fingerprint,
        "schema_version": 1,
        "scorer_version": SCORER_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "data": str(data_path.resolve()),
        "data_sha256": data_hash,
        "model": str(Path(args.model).resolve()),
        "tokenizer": str(Path(tokenizer_path).resolve()),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_prompts": len(prepared),
        "eval_indices": indices,
        "benchmark": args.benchmark,
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
        },
        "stop_tokens": stop_metadata,
        "engine_load_seconds": engine_ready - engine_started,
        "generation_start_epoch": generation_started,
        "generation_end_epoch": generation_finished,
        "generation_seconds": generation_finished - generation_started,
        "generated_token_count": sum(row["generated_token_count"] for row in predictions),
        "prediction_file": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
    }
    write_json_atomic(manifest_path, manifest)


def json_safe_stop_reason(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_manifests(stage_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for shard_index in range(4):
        path = stage_dir / f"shard_{shard_index:02d}_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("shard_index") != shard_index:
            raise ValueError(f"Invalid shard manifest: {path}")
        manifests.append(manifest)
    fingerprints = {manifest["data_sha256"] for manifest in manifests}
    models = {manifest["model"] for manifest in manifests}
    configs = {json.dumps(manifest["engine_config"], sort_keys=True) for manifest in manifests}
    if len(fingerprints) != 1 or len(models) != 1 or len(configs) != 1:
        raise ValueError("Shard manifests disagree on dataset, model, or engine configuration")
    return manifests


def throughput_metrics(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    generated = sum(int(manifest["generated_token_count"]) for manifest in manifests)
    start = min(float(manifest["generation_start_epoch"]) for manifest in manifests)
    end = max(float(manifest["generation_end_epoch"]) for manifest in manifests)
    makespan = end - start
    return {
        "aggregate_generated_tokens": generated,
        "four_worker_generation_makespan_seconds": makespan,
        "aggregate_generated_tokens_per_second": generated / makespan if makespan > 0 else 0.0,
        "engine_load_seconds_by_shard": [manifest["engine_load_seconds"] for manifest in manifests],
    }


def merge_benchmark(args: argparse.Namespace) -> None:
    manifests = load_manifests(Path(args.stage_dir))
    if not all(manifest.get("benchmark") for manifest in manifests):
        raise ValueError("A benchmark merge received a production shard")
    summary = {
        "status": "complete",
        "model": manifests[0]["model"],
        "data_sha256": manifests[0]["data_sha256"],
        "engine_config": manifests[0]["engine_config"],
        "num_prompts": sum(int(manifest["num_prompts"]) for manifest in manifests),
        **throughput_metrics(manifests),
    }
    write_json_atomic(Path(args.output), summary)


def select_tuning(args: argparse.Namespace) -> None:
    candidates: list[dict[str, Any]] = []
    for path_text in args.candidate:
        path = Path(path_text)
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "complete" and float(value.get("aggregate_generated_tokens_per_second", 0)) > 0:
            candidates.append(value)
    if not candidates:
        raise RuntimeError("No valid offline-vLLM tuning candidate completed")
    candidates.sort(
        key=lambda value: (
            float(value["aggregate_generated_tokens_per_second"]),
            not bool(value["engine_config"]["enforce_eager"]),
            -int(value["engine_config"]["max_num_seqs"]),
        ),
        reverse=True,
    )
    selected = candidates[0]
    write_json_atomic(
        Path(args.output),
        {
            "status": "selected",
            "selection_metric": "aggregate_generated_tokens_per_second",
            "selected_engine_config": selected["engine_config"],
            "selected_throughput": selected["aggregate_generated_tokens_per_second"],
            "candidates": candidates,
        },
    )


def merge_production(args: argparse.Namespace) -> None:
    stage_dir = Path(args.stage_dir)
    data_rows = read_jsonl(Path(args.data))
    manifests = load_manifests(stage_dir)
    if any(manifest.get("benchmark") for manifest in manifests):
        raise ValueError("A production merge received a benchmark shard")
    predictions: list[dict[str, Any]] = []
    for shard_index in range(4):
        path = stage_dir / f"shard_{shard_index:02d}_predictions.jsonl"
        shard_rows = read_jsonl(path)
        expected_indices = list(range(shard_index, len(data_rows), 4))
        if [int(row["eval_index"]) for row in shard_rows] != expected_indices:
            raise ValueError(f"Shard {shard_index} coverage/order mismatch")
        predictions.extend(shard_rows)
    predictions.sort(key=lambda row: int(row["eval_index"]))
    expected_ids = [str(row["problem_id"]) for row in data_rows]
    actual_ids = [str(row["problem_id"]) for row in predictions]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Fresh V2 predictions contain duplicate problem IDs")
    if actual_ids != expected_ids:
        raise ValueError("Fresh V2 predictions do not exactly match HARP benchmark order")
    prediction_path = Path(args.output_dir) / "predictions.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    metrics = {
        **metrics_from_predictions(predictions),
        **throughput_metrics(manifests),
        "data_sha256": sha256_file(Path(args.data)),
        "predictions_sha256": sha256_file(prediction_path),
        "model": manifests[0]["model"],
        "engine_config": manifests[0]["engine_config"],
        "stop_tokens": manifests[0]["stop_tokens"],
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
    }
    write_json_atomic(Path(args.output_dir) / "metrics.json", metrics)


def rescore(args: argparse.Namespace) -> None:
    data_rows = read_jsonl(Path(args.data))
    source_path = Path(args.predictions)
    source_rows = read_jsonl(source_path)
    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    duplicate_ids: set[str] = set()
    for occurrence, row in enumerate(source_rows):
        problem_id = str(row.get("problem_id"))
        if problem_id in latest:
            duplicate_ids.add(problem_id)
        latest[problem_id] = (occurrence, row)
    rescored: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    for eval_index, benchmark_row in enumerate(data_rows):
        problem_id = str(benchmark_row["problem_id"])
        if problem_id not in latest:
            raise ValueError(f"Historical predictions are missing HARP ID {problem_id}")
        _, old = latest[problem_id]
        text = old.get("generated_text", old.get("response", old.get("completion", "")))
        generated_count = int(old.get("generated_token_count", 0))
        max_tokens = int(old.get("max_tokens", args.max_tokens))
        finish_reason = old.get("finish_reason")
        row = {
            **old,
            "problem_id": problem_id,
            "eval_index": eval_index,
            "correct_answer": benchmark_row["correct_answer"],
            "generated_text": text,
            "generated_token_count": generated_count,
            "max_tokens": max_tokens,
            "finish_reason": finish_reason,
            "cap_hit": finish_reason == "length" or generated_count >= max_tokens,
            **score_answer(str(text), str(benchmark_row["correct_answer"])),
        }
        if "is_correct" in old and bool(old["is_correct"]) != bool(row["is_correct"]):
            flips.append({"problem_id": problem_id, "legacy_is_correct": bool(old["is_correct"]), "v2_is_correct": bool(row["is_correct"])})
        rescored.append(row)
    if len(rescored) != len(data_rows):
        raise ValueError("Historical rescore coverage mismatch")
    output_dir = Path(args.output_dir)
    if output_dir.name != "rescore_harp_answer_v2":
        raise ValueError("Historical V2 overlays must use a rescore_harp_answer_v2/ sibling directory")
    prediction_path = output_dir / "predictions.jsonl"
    write_jsonl_atomic(prediction_path, rescored)
    write_json_atomic(output_dir / "metrics.json", metrics_from_predictions(rescored))
    write_json_atomic(
        output_dir / "rescore_audit.json",
        {
            "scorer_version": SCORER_VERSION,
            "source_predictions": str(source_path.resolve()),
            "source_predictions_sha256": sha256_file(source_path),
            "source_artifact_duplicate_ids_latest_retained": sorted(duplicate_ids),
            "v2_predictions_sha256": sha256_file(prediction_path),
            "correctness_flips": flips,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--model", required=True)
    shard.add_argument("--tokenizer")
    shard.add_argument("--data", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--num-shards", type=int, default=4)
    shard.add_argument("--expected-samples", type=int, default=500)
    shard.add_argument("--max-model-len", type=int, default=32768)
    shard.add_argument("--max-response-len", type=int, default=31744)
    shard.add_argument("--max-num-seqs", type=int, required=True)
    shard.add_argument("--max-num-batched-tokens", type=int, default=16384)
    shard.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    shard.add_argument("--enforce-eager", action="store_true")
    shard.add_argument("--benchmark", action="store_true")
    shard.add_argument("--benchmark-total-size", type=int, default=64)
    shard.add_argument("--benchmark-max-tokens", type=int, default=4096)
    shard.add_argument("--resume", action="store_true")
    shard.add_argument("--ready-file")

    benchmark_merge = subparsers.add_parser("merge-benchmark")
    benchmark_merge.add_argument("--stage-dir", required=True)
    benchmark_merge.add_argument("--output", required=True)
    selection = subparsers.add_parser("select-tuning")
    selection.add_argument("--candidate", action="append", required=True)
    selection.add_argument("--output", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--stage-dir", required=True)
    merge.add_argument("--data", required=True)
    merge.add_argument("--output-dir", required=True)
    historical = subparsers.add_parser("rescore")
    historical.add_argument("--data", required=True)
    historical.add_argument("--predictions", required=True)
    historical.add_argument("--output-dir", required=True)
    historical.add_argument("--max-tokens", type=int, default=31744)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "shard":
        run_shard(args)
    elif args.command == "merge-benchmark":
        merge_benchmark(args)
    elif args.command == "select-tuning":
        select_tuning(args)
    elif args.command == "merge":
        merge_production(args)
    elif args.command == "rescore":
        rescore(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
