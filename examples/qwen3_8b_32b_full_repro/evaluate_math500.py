#!/usr/bin/env python3
"""Read-only, resumable, four-shard MATH-500 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen3_8b_32b_full_repro.math500_scorer import SCORER_VERSION, score_response


PROMPT_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shard = sub.add_parser("shard")
    shard.add_argument("--model", required=True)
    shard.add_argument("--stage", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--num-shards", type=int, default=4)
    shard.add_argument("--dataset", default=os.environ.get("MATH500_DATASET", "HuggingFaceH4/MATH-500"))
    shard.add_argument("--revision", default=os.environ.get("MATH500_REVISION"), required=not bool(os.environ.get("MATH500_REVISION")))
    shard.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    shard.add_argument("--temperature", type=float, default=0.8)
    shard.add_argument("--top-p", type=float, default=0.7)
    shard.add_argument("--top-k", type=int, default=-1)
    shard.add_argument("--max-new-tokens", type=int, default=16384)
    shard.add_argument("--safety-margin", type=int, default=64)
    shard.add_argument("--seed", type=int, default=1234)
    shard.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    shard.add_argument("--max-num-seqs", type=int, default=24)
    shard.add_argument("--max-num-batched-tokens", type=int, default=16384)
    shard.add_argument("--stop-token-ids", type=int, nargs="+", default=[151643, 151645])
    shard.add_argument("--resume", action="store_true")

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--stage", required=True)
    aggregate.add_argument("--output-dir", required=True)
    aggregate.add_argument("--expected", type=int, default=500)
    aggregate.add_argument("--bootstrap-seed", type=int, default=1234)
    aggregate.add_argument("--compare-to")
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


def dataset_rows(dataset_name: str, revision: str, cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, revision=revision, split="test", cache_dir=cache_dir)
    if len(dataset) != 500:
        raise RuntimeError(f"Pinned MATH-500 revision must contain exactly 500 original rows, found {len(dataset)}")
    return [dict(row) for row in dataset]


def model_context(config: Any) -> int:
    value = int(getattr(config, "max_position_embeddings", 0) or 0)
    if value <= 0:
        raise RuntimeError("Model config does not define a positive max_position_embeddings")
    return value


def policy(args: argparse.Namespace, context: int) -> dict[str, Any]:
    return {
        "model": str(Path(args.model).resolve()),
        "dataset": args.dataset,
        "revision": args.revision,
        "dataset_policy": "read_only_original_500_row_order",
        "prompt_instruction": PROMPT_INSTRUCTION,
        "enable_thinking": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "n": 1,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "model_context": context,
        "safety_margin": args.safety_margin,
        "stop_token_ids": sorted(set(args.stop_token_ids)),
        "scorer": SCORER_VERSION,
    }


def policy_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def result_path(output_dir: Path, index: int) -> Path:
    return output_dir / "problems" / f"problem_{index:04d}.json"


def compatible(path: Path, expected_policy_hash: str, stage: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return value.get("policy_sha256") == expected_policy_hash and value.get("stage") == stage


def run_shard(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    rows = dataset_rows(args.dataset, args.revision, args.cache_dir)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    context = model_context(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    expected_tokens = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
    for token, expected_id in expected_tokens.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected_id:
            raise RuntimeError(f"Tokenizer mismatch for {token}: expected {expected_id}, found {actual}")
    if set(expected_tokens.values()) - set(args.stop_token_ids):
        raise RuntimeError("Evaluation stop IDs must include EOS and <|im_end|>")
    generation_policy = policy(args, context)
    fingerprint = policy_hash(generation_policy)
    output_dir = Path(args.output_dir)
    assigned = [(idx, row) for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    pending: list[tuple[int, dict[str, Any], list[int], str, int]] = []
    for index, row in assigned:
        path = result_path(output_dir, index)
        if path.exists() and args.resume and compatible(path, fingerprint, args.stage):
            continue
        if path.exists():
            raise RuntimeError(f"Refusing incompatible or non-resume output: {path}")
        problem = str(row.get("problem", row.get("question", "")))
        user = problem.rstrip() + "\n\n" + PROMPT_INSTRUCTION
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer.encode(rendered, add_special_tokens=False)
        available = context - len(input_ids) - args.safety_margin
        effective_cap = max(1, min(args.max_new_tokens, available))
        pending.append((index, row, input_ids, rendered, effective_cap))
    print(f"EVAL_SHARD stage={args.stage} shard={args.shard_index}/{args.num_shards} pending={len(pending)} policy={fingerprint}", flush=True)
    if not pending:
        return
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        safetensors_load_strategy="prefetch",
    )
    prompts = [TokensPrompt(prompt_token_ids=item[2]) for item in pending]
    sampling = [
        SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=item[4],
            seed=args.seed + item[0],
            stop_token_ids=sorted(set(args.stop_token_ids)),
            ignore_eos=True,
            detokenize=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            include_stop_str_in_output=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        for item in pending
    ]
    outputs = engine.generate(prompts, sampling_params=sampling, use_tqdm=True)
    if len(outputs) != len(pending):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(pending)} prompts")
    for (index, row, input_ids, rendered, cap), output in zip(pending, outputs, strict=True):
        completion = output.outputs[0]
        token_ids = [int(value) for value in (completion.token_ids or [])]
        finish_reason = str(completion.finish_reason or "")
        raw_stop = getattr(completion, "stop_reason", None)
        stop_token_id = int(raw_stop) if isinstance(raw_stop, int) or (isinstance(raw_stop, str) and raw_stop.isdigit()) else None
        if stop_token_id is None and finish_reason == "stop":
            # vLLM may omit stop_reason for the tokenizer-native EOS path. The
            # same EOS ID is also in stop_token_ids, so this remains exact.
            stop_token_id = int(tokenizer.eos_token_id)
        response = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        cap_hit = finish_reason == "length" or len(token_ids) >= cap
        gold = str(row.get("answer", row.get("solution", row.get("label", ""))))
        trace = score_response(
            response,
            gold,
            finish_reason=finish_reason,
            stop_token_id=stop_token_id,
            cap_hit=cap_hit,
            problem_id=index,
        )
        result = {
            "artifact_schema_version": 1,
            "stage": args.stage,
            "eval_index": index,
            "problem": str(row.get("problem", row.get("question", ""))),
            "gold_answer": gold,
            "rendered_prompt": rendered,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "input_token_ids": input_ids,
            "input_tokens": len(input_ids),
            "configured_max_new_tokens": args.max_new_tokens,
            "effective_max_new_tokens": cap,
            "generated_token_ids": token_ids,
            "generated_tokens": len(token_ids),
            "response": response,
            "finish_reason": finish_reason,
            "stop_token_id": stop_token_id,
            "stop_condition": (
                "eos" if stop_token_id == tokenizer.eos_token_id else
                "im_end" if stop_token_id == tokenizer.convert_tokens_to_ids("<|im_end|>") else
                "length_cap" if cap_hit else finish_reason
            ),
            "cap_hit": cap_hit,
            "seed": args.seed + index,
            "policy": generation_policy,
            "policy_sha256": fingerprint,
            "scorer_trace": trace,
        }
        atomic_text(result_path(output_dir, index), json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"EVAL_SAMPLE_COMMIT stage={args.stage} index={index} correct={int(trace['is_correct'])} path={result_path(output_dir, index)}", flush=True)
    if hasattr(engine, "shutdown"):
        engine.shutdown()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(correct: list[bool], seed: int, samples: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    n = len(correct)
    estimates = [sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return [percentile([round(value * 1_000_000) for value in estimates], 0.025) / 1_000_000, percentile([round(value * 1_000_000) for value in estimates], 0.975) / 1_000_000]


def checkpoint_manifest(model: str) -> dict[str, Any]:
    root = Path(model)
    files = []
    for path in sorted(root.glob("*.json")) + sorted(root.glob("*.safetensors")) + sorted(root.glob("*.bin")):
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"path": str(root.resolve()), "files": files, "manifest_sha256": hashlib.sha256(canonical).hexdigest()}


def binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    probability = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2 * probability)


def aggregate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    paths = sorted((output_dir / "problems").glob("problem_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    indices = [int(row["eval_index"]) for row in rows]
    if len(rows) != args.expected or indices != list(range(args.expected)):
        missing = sorted(set(range(args.expected)) - set(indices))[:30]
        raise RuntimeError(f"Expected original indices 0..{args.expected - 1}, found {len(rows)} rows; missing={missing}")
    fingerprints = {row["policy_sha256"] for row in rows}
    stages = {row["stage"] for row in rows}
    if len(fingerprints) != 1 or stages != {args.stage}:
        raise RuntimeError(f"Mixed evaluation artifacts: policies={fingerprints} stages={stages}")
    correct = [bool(row["scorer_trace"]["is_correct"]) for row in rows]
    lengths = [int(row["generated_tokens"]) for row in rows]
    metrics: dict[str, Any] = {
        "stage": args.stage,
        "scorer_version": SCORER_VERSION,
        "policy_sha256": next(iter(fingerprints)),
        "correct": sum(correct),
        "total": len(rows),
        "pass_at_1": sum(correct) / len(rows),
        "bootstrap_95_ci": bootstrap_ci(correct, args.bootstrap_seed),
        "parseable": sum(row["scorer_trace"]["official_extracted_candidate"] is not None for row in rows),
        "parser_verifier_failures": sum(bool(row["scorer_trace"]["parser_verifier_exception"]) for row in rows),
        "length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "cap_hits": sum(bool(row["cap_hit"]) for row in rows),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "stop_tokens": dict(Counter(str(row["stop_token_id"]) for row in rows)),
        "think_closed": sum(bool(row["scorer_trace"]["think_closure"]) for row in rows),
        "eligible_final_answers": sum(row["scorer_trace"]["official_extracted_candidate"] is not None for row in rows),
        "answer_anywhere": sum(row["scorer_trace"]["answer_anywhere_diagnostic"] is not None for row in rows),
        "replacements": sum(int(row["scorer_trace"]["answer_replacement_count"]) for row in rows),
        "model_checkpoint": checkpoint_manifest(str(rows[0]["policy"].get("model", ""))) if rows[0]["policy"].get("model") else None,
    }
    if args.compare_to:
        baseline = json.loads(Path(args.compare_to).read_text())
        base_by_index = {int(row["eval_index"]): bool(row["is_correct"]) for row in baseline["per_problem"]}
        wrong_correct = sum(not base_by_index[i] and correct[i] for i in range(len(rows)))
        correct_wrong = sum(base_by_index[i] and not correct[i] for i in range(len(rows)))
        metrics["paired"] = {
            "wrong_to_correct": wrong_correct,
            "correct_to_wrong": correct_wrong,
            "mcnemar_exact_p": binomial_two_sided(min(wrong_correct, correct_wrong), wrong_correct + correct_wrong),
        }
    metrics["per_problem"] = [
        {
            "eval_index": row["eval_index"],
            "is_correct": row["scorer_trace"]["is_correct"],
            "candidate": row["scorer_trace"]["official_extracted_candidate"],
            "candidate_sha256": row["scorer_trace"]["official_candidate_sha256"],
            "generated_text_sha256": row["scorer_trace"]["generated_text_sha256"],
        }
        for row in rows
    ]
    raw_artifacts = [
        {"eval_index": int(row["eval_index"]), "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for row, path in zip(rows, paths, strict=True)
    ]
    metrics["raw_artifact_manifest_sha256"] = hashlib.sha256(
        json.dumps(raw_artifacts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metrics["raw_artifacts"] = raw_artifacts
    # Resolve the exact checkpoint from the frozen per-row sampling policy.
    model_path = str(rows[0].get("model", rows[0]["policy"].get("model", "")))
    if model_path:
        metrics["model_checkpoint"] = checkpoint_manifest(model_path)
    atomic_text(output_dir / "summary.json", json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_text(output_dir / "per_problem.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    atomic_text(
        output_dir / "summary.md",
        f"# {args.stage} MATH-500\n\n- Pass@1: {metrics['correct']}/{metrics['total']} ({metrics['pass_at_1']:.2%})\n- 95% bootstrap CI: {metrics['bootstrap_95_ci']}\n- Mean/median/p95/max tokens: {metrics['length']['mean']:.1f}/{metrics['length']['median']}/{metrics['length']['p95']}/{metrics['length']['max']}\n- Cap hits: {metrics['cap_hits']}\n- Think closed: {metrics['think_closed']}\n- Eligible answers: {metrics['eligible_final_answers']}\n",
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_problem"}, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "shard":
        run_shard(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
