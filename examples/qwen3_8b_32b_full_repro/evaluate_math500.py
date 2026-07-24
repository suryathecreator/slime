#!/usr/bin/env python3
"""Read-only, resumable MATH-500 evaluation and exact cap-hit proxy."""

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
    shard.add_argument("--target-context", type=int, default=32768)
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

    cap_rerun = sub.add_parser("cap-rerun-shard")
    cap_rerun.add_argument("--model", required=True)
    cap_rerun.add_argument("--stage", required=True)
    cap_rerun.add_argument("--source-dir", required=True)
    cap_rerun.add_argument("--output-dir", required=True)
    cap_rerun.add_argument("--shard-index", type=int, required=True)
    cap_rerun.add_argument("--num-shards", type=int, default=4)
    cap_rerun.add_argument("--expected", type=int, default=500)
    cap_rerun.add_argument("--target-context", type=int, default=32768)
    cap_rerun.add_argument("--safety-margin", type=int, default=64)
    cap_rerun.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    cap_rerun.add_argument("--max-num-seqs", type=int, default=24)
    cap_rerun.add_argument("--max-num-batched-tokens", type=int, default=16384)
    cap_rerun.add_argument("--resume", action="store_true")

    cap_aggregate = sub.add_parser("aggregate-cap-proxy")
    cap_aggregate.add_argument("--stage", required=True)
    cap_aggregate.add_argument("--source-dir", required=True)
    cap_aggregate.add_argument("--output-dir", required=True)
    cap_aggregate.add_argument("--expected", type=int, default=500)
    cap_aggregate.add_argument("--target-context", type=int, default=32768)
    cap_aggregate.add_argument("--safety-margin", type=int, default=64)
    cap_aggregate.add_argument("--bootstrap-seed", type=int, default=1234)
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


def policy(args: argparse.Namespace, native_context: int) -> dict[str, Any]:
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
        "response_budget_mode": "min(configured_max_new_tokens, target_context - rendered_prompt_tokens - safety_margin)",
        "target_total_context_tokens": args.target_context,
        "model_native_context_tokens": native_context,
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


def dynamic_response_budget(input_tokens: int, target_context: int, safety_margin: int) -> int:
    if input_tokens < 1:
        raise RuntimeError(f"Input token count must be positive, found {input_tokens}")
    if target_context < 1 or safety_margin < 0:
        raise RuntimeError(
            f"Invalid context budget: target_context={target_context} safety_margin={safety_margin}"
        )
    available = target_context - input_tokens - safety_margin
    if available < 1:
        raise RuntimeError(
            f"Prompt leaves no response budget: {target_context} - {input_tokens} - {safety_margin} = {available}"
        )
    return available


def require_exact_prefix(source_ids: list[int], rerun_ids: list[int], *, index: int) -> None:
    if len(rerun_ids) < len(source_ids):
        raise RuntimeError(
            f"Cap proxy prefix mismatch at index {index}: rerun has {len(rerun_ids)} tokens, "
            f"source has {len(source_ids)}"
        )
    if rerun_ids[: len(source_ids)] != source_ids:
        mismatch = next(
            offset
            for offset, (source, rerun) in enumerate(zip(source_ids, rerun_ids, strict=False))
            if source != rerun
        )
        raise RuntimeError(
            f"Cap proxy prefix mismatch at index {index}, token offset {mismatch}: "
            f"source={source_ids[mismatch]} rerun={rerun_ids[mismatch]}"
        )


def load_source_evaluation(
    source_dir: Path, expected: int, *, verify_raw_hashes: bool
) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any], str]:
    summary_path = source_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing source summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    paths = sorted((source_dir / "problems").glob("problem_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    indices = [int(row["eval_index"]) for row in rows]
    if len(rows) != expected or indices != list(range(expected)):
        missing = sorted(set(range(expected)) - set(indices))[:30]
        raise RuntimeError(
            f"Expected source indices 0..{expected - 1}, found {len(rows)} rows; missing={missing}"
        )
    stages = {str(row["stage"]) for row in rows}
    fingerprints = {str(row["policy_sha256"]) for row in rows}
    if len(stages) != 1 or len(fingerprints) != 1:
        raise RuntimeError(f"Mixed source artifacts: stages={stages} policies={fingerprints}")
    source_stage = next(iter(stages))
    source_policy = next(iter(fingerprints))
    if summary.get("stage") != source_stage or summary.get("policy_sha256") != source_policy:
        raise RuntimeError("Source summary stage or policy differs from its raw problem artifacts")
    if int(summary.get("total", -1)) != expected:
        raise RuntimeError(f"Source summary expected {expected} rows, found {summary.get('total')}")
    if int(summary.get("cap_hits", -1)) != sum(bool(row["cap_hit"]) for row in rows):
        raise RuntimeError("Source summary cap-hit count differs from raw problem artifacts")
    if verify_raw_hashes:
        manifest = {int(item["eval_index"]): item for item in summary.get("raw_artifacts", [])}
        if set(manifest) != set(range(expected)):
            raise RuntimeError("Source summary raw-artifact manifest is incomplete")
        for row, path in zip(rows, paths, strict=True):
            item = manifest[int(row["eval_index"])]
            actual = sha256_file(path)
            if item.get("sha256") != actual:
                raise RuntimeError(f"Source raw artifact hash changed: {path}")
    return rows, paths, summary, sha256_file(summary_path)


def cap_proxy_policy(
    *,
    model: str,
    source_summary_sha256: str,
    source_policy_sha256: str,
    source_stage: str,
    target_context: int,
    safety_margin: int,
) -> dict[str, Any]:
    return {
        "diagnostic": "exact_cap_hit_only_32k_total_context_proxy_v1",
        "model": str(Path(model).resolve()),
        "source_stage": source_stage,
        "source_summary_sha256": source_summary_sha256,
        "source_policy_sha256": source_policy_sha256,
        "source_selection": "cap_hit_true_only",
        "natural_stop_policy": "reuse_saved_source_generation_without_regeneration",
        "replay_policy": "rerun_original_saved_input_tokens_with_original_per_problem_seed",
        "required_prefix_check": "rerun_generated_token_ids[:source_generated_tokens] == source_generated_token_ids",
        "target_total_context_tokens": target_context,
        "response_budget": f"{target_context} - rendered_prompt_tokens - {safety_margin}",
        "context_accounting": "rendered prompt tokens + generated response tokens + reserved safety tokens",
        "safety_margin": safety_margin,
        "scorer": SCORER_VERSION,
    }


def cap_proxy_compatible(
    path: Path,
    *,
    expected_policy_hash: str,
    stage: str,
    source_sha256: str,
    source_ids: list[int],
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
        rerun_ids = [int(token) for token in value["generated_token_ids"]]
        require_exact_prefix(source_ids, rerun_ids, index=int(value["eval_index"]))
    except Exception:
        return False
    return (
        value.get("policy_sha256") == expected_policy_hash
        and value.get("stage") == stage
        and value.get("source_artifact", {}).get("sha256") == source_sha256
        and value.get("prefix_validation", {}).get("exact_match") is True
    )


def run_shard(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    rows = dataset_rows(args.dataset, args.revision, args.cache_dir)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    native_context = model_context(config)
    if native_context < args.target_context:
        raise RuntimeError(
            f"Model native context {native_context} is below target context {args.target_context}"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    expected_tokens = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
    for token, expected_id in expected_tokens.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected_id:
            raise RuntimeError(f"Tokenizer mismatch for {token}: expected {expected_id}, found {actual}")
    if set(expected_tokens.values()) - set(args.stop_token_ids):
        raise RuntimeError("Evaluation stop IDs must include EOS and <|im_end|>")
    generation_policy = policy(args, native_context)
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
        available = dynamic_response_budget(
            len(input_ids), args.target_context, args.safety_margin
        )
        effective_cap = min(args.max_new_tokens, available)
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
        max_model_len=args.target_context,
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


def run_cap_rerun_shard(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    source_dir = Path(args.source_dir)
    source_rows, source_paths, source_summary, source_summary_sha256 = load_source_evaluation(
        source_dir, args.expected, verify_raw_hashes=False
    )
    source_stage = str(source_summary["stage"])
    source_policy_sha256 = str(source_summary["policy_sha256"])
    source_model = str(source_rows[0]["policy"]["model"])
    if Path(source_model).resolve() != Path(args.model).resolve():
        raise RuntimeError(f"Proxy model differs from source: {args.model} != {source_model}")
    source_scorers = {str(row["policy"].get("scorer")) for row in source_rows}
    if source_scorers != {SCORER_VERSION}:
        raise RuntimeError(f"Source scorer differs from current scorer: {source_scorers} != {SCORER_VERSION}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    native_context = model_context(config)
    if native_context < args.target_context:
        raise RuntimeError(
            f"Model native context {native_context} is below proxy target {args.target_context}"
        )
    generation_policy = cap_proxy_policy(
        model=args.model,
        source_summary_sha256=source_summary_sha256,
        source_policy_sha256=source_policy_sha256,
        source_stage=source_stage,
        target_context=args.target_context,
        safety_margin=args.safety_margin,
    )
    fingerprint = policy_hash(generation_policy)
    output_dir = Path(args.output_dir)
    selected = [
        (row, path)
        for row, path in zip(source_rows, source_paths, strict=True)
        if bool(row["cap_hit"])
        and int(row["eval_index"]) % args.num_shards == args.shard_index
    ]
    source_manifest = {
        int(item["eval_index"]): item for item in source_summary.get("raw_artifacts", [])
    }
    if set(source_manifest) != set(range(args.expected)):
        raise RuntimeError("Source summary raw-artifact manifest is incomplete")
    pending: list[tuple[dict[str, Any], Path, str, int]] = []
    for source, source_path in selected:
        index = int(source["eval_index"])
        source_ids = [int(token) for token in source["generated_token_ids"]]
        if len(source_ids) != int(source["generated_tokens"]):
            raise RuntimeError(f"Source generated-token count mismatch at index {index}")
        if int(source.get("seed", -1)) != int(source["policy"].get("seed", 0)) + index:
            raise RuntimeError(f"Source per-problem seed mismatch at index {index}")
        source_sha256 = sha256_file(source_path)
        if source_manifest[index].get("sha256") != source_sha256:
            raise RuntimeError(f"Source raw artifact hash changed: {source_path}")
        target_path = result_path(output_dir, index)
        if target_path.exists() and args.resume and cap_proxy_compatible(
            target_path,
            expected_policy_hash=fingerprint,
            stage=args.stage,
            source_sha256=source_sha256,
            source_ids=source_ids,
        ):
            continue
        if target_path.exists():
            raise RuntimeError(f"Refusing incompatible or non-resume proxy output: {target_path}")
        budget = dynamic_response_budget(
            int(source["input_tokens"]), args.target_context, args.safety_margin
        )
        if budget <= len(source_ids):
            raise RuntimeError(
                f"32K proxy budget does not extend source row {index}: budget={budget} source={len(source_ids)}"
            )
        pending.append((source, source_path, source_sha256, budget))
    print(
        f"CAP_PROXY_SHARD stage={args.stage} source={source_stage} "
        f"shard={args.shard_index}/{args.num_shards} selected={len(selected)} "
        f"pending={len(pending)} policy={fingerprint}",
        flush=True,
    )
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    expected_tokens = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
    for token, expected_id in expected_tokens.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected_id:
            raise RuntimeError(f"Tokenizer mismatch for {token}: expected {expected_id}, found {actual}")
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.target_context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        safetensors_load_strategy="prefetch",
    )
    prompts = [TokensPrompt(prompt_token_ids=[int(token) for token in item[0]["input_token_ids"]]) for item in pending]
    sampling = []
    for source, _, _, budget in pending:
        source_policy = source["policy"]
        sampling.append(
            SamplingParams(
                n=1,
                temperature=float(source_policy["temperature"]),
                top_p=float(source_policy["top_p"]),
                top_k=int(source_policy["top_k"]),
                max_tokens=budget,
                seed=int(source["seed"]),
                stop_token_ids=sorted({int(token) for token in source_policy["stop_token_ids"]}),
                ignore_eos=True,
                detokenize=False,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                include_stop_str_in_output=True,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
        )
    outputs = engine.generate(prompts, sampling_params=sampling, use_tqdm=True)
    if len(outputs) != len(pending):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(pending)} proxy prompts")
    for (source, source_path, source_sha256, budget), output in zip(pending, outputs, strict=True):
        index = int(source["eval_index"])
        completion = output.outputs[0]
        token_ids = [int(value) for value in (completion.token_ids or [])]
        source_ids = [int(token) for token in source["generated_token_ids"]]
        # This is the publication gate. A rerun that does not reproduce the
        # saved sampled prefix is not a continuation proxy and is never scored.
        require_exact_prefix(source_ids, token_ids, index=index)
        finish_reason = str(completion.finish_reason or "")
        raw_stop = getattr(completion, "stop_reason", None)
        stop_token_id = (
            int(raw_stop)
            if isinstance(raw_stop, int) or (isinstance(raw_stop, str) and raw_stop.isdigit())
            else None
        )
        if stop_token_id is None and finish_reason == "stop":
            stop_token_id = int(tokenizer.eos_token_id)
        response = tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        cap_hit = finish_reason == "length" or len(token_ids) >= budget
        trace = score_response(
            response,
            str(source["gold_answer"]),
            finish_reason=finish_reason,
            stop_token_id=stop_token_id,
            cap_hit=cap_hit,
            problem_id=index,
        )
        result = {
            "artifact_schema_version": 2,
            "artifact_kind": "exact_cap_hit_only_32k_total_context_proxy",
            "stage": args.stage,
            "eval_index": index,
            "problem": source["problem"],
            "gold_answer": source["gold_answer"],
            "rendered_prompt": source["rendered_prompt"],
            "rendered_prompt_sha256": source["rendered_prompt_sha256"],
            "input_token_ids": source["input_token_ids"],
            "input_tokens": int(source["input_tokens"]),
            "response_budget_mode": "dynamic_total_context",
            "target_total_context_tokens": args.target_context,
            "safety_margin": args.safety_margin,
            "effective_max_new_tokens": budget,
            "generated_token_ids": token_ids,
            "generated_tokens": len(token_ids),
            "response": response,
            "finish_reason": finish_reason,
            "stop_token_id": stop_token_id,
            "stop_condition": (
                "eos"
                if stop_token_id == tokenizer.eos_token_id
                else "im_end"
                if stop_token_id == tokenizer.convert_tokens_to_ids("<|im_end|>")
                else "length_cap"
                if cap_hit
                else finish_reason
            ),
            "cap_hit": cap_hit,
            "seed": int(source["seed"]),
            "policy": generation_policy,
            "policy_sha256": fingerprint,
            "source_artifact": {
                "path": str(source_path),
                "sha256": source_sha256,
                "stage": source_stage,
                "policy_sha256": source_policy_sha256,
                "generated_tokens": len(source_ids),
                "cap_hit": True,
            },
            "prefix_validation": {
                "exact_match": True,
                "matched_tokens": len(source_ids),
                "required": True,
            },
            "scorer_trace": trace,
        }
        target_path = result_path(output_dir, index)
        atomic_text(target_path, json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"CAP_PROXY_SAMPLE_COMMIT stage={args.stage} index={index} "
            f"source_tokens={len(source_ids)} proxy_tokens={len(token_ids)} "
            f"correct={int(trace['is_correct'])} path={target_path}",
            flush=True,
        )
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


def aggregate_cap_proxy(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    source_rows, source_paths, source_summary, source_summary_sha256 = load_source_evaluation(
        source_dir, args.expected, verify_raw_hashes=True
    )
    source_stage = str(source_summary["stage"])
    source_policy_sha256 = str(source_summary["policy_sha256"])
    source_model = str(source_rows[0]["policy"]["model"])
    expected_policy = cap_proxy_policy(
        model=source_model,
        source_summary_sha256=source_summary_sha256,
        source_policy_sha256=source_policy_sha256,
        source_stage=source_stage,
        target_context=args.target_context,
        safety_margin=args.safety_margin,
    )
    expected_policy_sha256 = policy_hash(expected_policy)
    selected_indices = [int(row["eval_index"]) for row in source_rows if bool(row["cap_hit"])]
    rerun_paths = sorted((output_dir / "problems").glob("problem_*.json"))
    reruns = [json.loads(path.read_text()) for path in rerun_paths]
    rerun_indices = [int(row["eval_index"]) for row in reruns]
    if rerun_indices != selected_indices:
        missing = sorted(set(selected_indices) - set(rerun_indices))[:30]
        extra = sorted(set(rerun_indices) - set(selected_indices))[:30]
        raise RuntimeError(
            f"Proxy must contain exactly the source cap-hit rows; "
            f"expected={len(selected_indices)} found={len(reruns)} missing={missing} extra={extra}"
        )
    rerun_by_index = {int(row["eval_index"]): (row, path) for row, path in zip(reruns, rerun_paths, strict=True)}
    composite: list[tuple[dict[str, Any], Path, str]] = []
    for source, source_path in zip(source_rows, source_paths, strict=True):
        index = int(source["eval_index"])
        if not bool(source["cap_hit"]):
            composite.append((source, source_path, "source_natural_stop"))
            continue
        rerun, rerun_path = rerun_by_index[index]
        source_sha256 = sha256_file(source_path)
        if rerun.get("stage") != args.stage or rerun.get("policy_sha256") != expected_policy_sha256:
            raise RuntimeError(f"Incompatible proxy policy or stage at index {index}")
        if rerun.get("source_artifact", {}).get("sha256") != source_sha256:
            raise RuntimeError(f"Proxy source hash mismatch at index {index}")
        source_ids = [int(token) for token in source["generated_token_ids"]]
        rerun_ids = [int(token) for token in rerun["generated_token_ids"]]
        require_exact_prefix(source_ids, rerun_ids, index=index)
        if rerun.get("prefix_validation", {}).get("exact_match") is not True:
            raise RuntimeError(f"Proxy prefix gate was not recorded as passed at index {index}")
        expected_budget = dynamic_response_budget(
            int(source["input_tokens"]), args.target_context, args.safety_margin
        )
        if int(rerun.get("effective_max_new_tokens", -1)) != expected_budget:
            raise RuntimeError(f"Proxy dynamic budget mismatch at index {index}")
        if int(rerun.get("seed", -1)) != int(source["seed"]):
            raise RuntimeError(f"Proxy seed mismatch at index {index}")
        composite.append((rerun, rerun_path, "cap_rerun"))

    rows = [item[0] for item in composite]
    paths = [item[1] for item in composite]
    provenance = [item[2] for item in composite]
    correct = [bool(row["scorer_trace"]["is_correct"]) for row in rows]
    source_correct = [bool(row["scorer_trace"]["is_correct"]) for row in source_rows]
    lengths = [int(row["generated_tokens"]) for row in rows]
    selected_source_correct = sum(source_correct[index] for index in selected_indices)
    selected_proxy_correct = sum(correct[index] for index in selected_indices)
    selected_resolved = sum(not bool(rows[index]["cap_hit"]) for index in selected_indices)
    wrong_correct = sum(not before and after for before, after in zip(source_correct, correct, strict=True))
    correct_wrong = sum(before and not after for before, after in zip(source_correct, correct, strict=True))
    metrics: dict[str, Any] = {
        "stage": args.stage,
        "diagnostic_kind": "exact_cap_hit_only_32k_total_context_proxy_v1",
        "interpretation": (
            "Exact proxy for the corresponding 32K-total-context evaluation because natural stops are reused "
            "and every regenerated cap-hit row passed a token-for-token source-prefix check."
        ),
        "not_a_primary_protocol_result": True,
        "scorer_version": SCORER_VERSION,
        "policy_sha256": expected_policy_sha256,
        "source": {
            "stage": source_stage,
            "directory": str(source_dir),
            "summary_sha256": source_summary_sha256,
            "policy_sha256": source_policy_sha256,
            "primary_correct": int(source_summary["correct"]),
            "primary_total": int(source_summary["total"]),
            "primary_pass_at_1": float(source_summary["pass_at_1"]),
            "primary_cap_hits": int(source_summary["cap_hits"]),
        },
        "context_budget": {
            "source": "min(16384, 32768 - rendered_prompt_tokens - 64)",
            "proxy": f"{args.target_context} - rendered_prompt_tokens - {args.safety_margin}",
            "accounting": "rendered prompt tokens + generated response tokens + reserved safety tokens",
            "rendered_prompt_includes": "chat-template special tokens, problem, instruction, and assistant prefix",
            "generated_response_includes": "reasoning, think tags, answer text, and any emitted stop token",
            "hidden_reasoning_stream": False,
        },
        "exactness_gate": {
            "source_cap_hits_selected": len(selected_indices),
            "prefixes_checked": len(selected_indices),
            "prefixes_exact": len(selected_indices),
            "prefix_mismatches": 0,
            "natural_stop_rows_reused": args.expected - len(selected_indices),
            "publication_refused_on_any_prefix_mismatch": True,
        },
        "correct": sum(correct),
        "total": len(rows),
        "pass_at_1": sum(correct) / len(rows),
        "bootstrap_95_ci": bootstrap_ci(correct, args.bootstrap_seed),
        "paired_to_source": {
            "wrong_to_correct": wrong_correct,
            "correct_to_wrong": correct_wrong,
            "mcnemar_exact_p": binomial_two_sided(
                min(wrong_correct, correct_wrong), wrong_correct + correct_wrong
            ),
        },
        "cap_hit_reruns": {
            "selected": len(selected_indices),
            "source_correct": selected_source_correct,
            "proxy_correct": selected_proxy_correct,
            "resolved_to_natural_stop": selected_resolved,
            "remaining_cap_hits": sum(bool(rows[index]["cap_hit"]) for index in selected_indices),
        },
        "parseable": sum(
            row["scorer_trace"]["official_extracted_candidate"] is not None for row in rows
        ),
        "parser_verifier_failures": sum(
            bool(row["scorer_trace"]["parser_verifier_exception"]) for row in rows
        ),
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
        "eligible_final_answers": sum(
            row["scorer_trace"]["official_extracted_candidate"] is not None for row in rows
        ),
        "answer_anywhere": sum(
            row["scorer_trace"]["answer_anywhere_diagnostic"] is not None for row in rows
        ),
        "replacements": sum(
            int(row["scorer_trace"]["answer_replacement_count"]) for row in rows
        ),
        "model_checkpoint": checkpoint_manifest(source_model),
    }
    metrics["per_problem"] = [
        {
            "eval_index": int(row["eval_index"]),
            "provenance": row_provenance,
            "is_correct": bool(row["scorer_trace"]["is_correct"]),
            "candidate": row["scorer_trace"]["official_extracted_candidate"],
            "candidate_sha256": row["scorer_trace"]["official_candidate_sha256"],
            "generated_text_sha256": row["scorer_trace"]["generated_text_sha256"],
            "generated_tokens": int(row["generated_tokens"]),
            "finish_reason": row["finish_reason"],
            "cap_hit": bool(row["cap_hit"]),
        }
        for row, row_provenance in zip(rows, provenance, strict=True)
    ]
    raw_artifacts = [
        {
            "eval_index": int(row["eval_index"]),
            "provenance": row_provenance,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for row, path, row_provenance in zip(rows, paths, provenance, strict=True)
    ]
    metrics["raw_artifact_manifest_sha256"] = hashlib.sha256(
        json.dumps(raw_artifacts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metrics["raw_artifacts"] = raw_artifacts
    atomic_text(
        output_dir / "summary.json",
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_text(
        output_dir / "per_problem.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metrics["per_problem"]),
    )
    atomic_text(
        output_dir / "summary.md",
        f"# {args.stage} MATH-500 cap-hit-only 32K proxy\n\n"
        f"- Source primary result: {source_summary['correct']}/{source_summary['total']} "
        f"({source_summary['pass_at_1']:.2%}) with {source_summary['cap_hits']} cap hits\n"
        f"- 32K-context proxy: {metrics['correct']}/{metrics['total']} ({metrics['pass_at_1']:.2%})\n"
        f"- Exact saved-prefix checks: {len(selected_indices)}/{len(selected_indices)}\n"
        f"- Reruns resolved to natural stop: {selected_resolved}/{len(selected_indices)}\n"
        f"- Remaining 32K cap hits: {metrics['cap_hits']}\n"
        f"- Mean/median/p95/max composite tokens: {metrics['length']['mean']:.1f}/"
        f"{metrics['length']['median']}/{metrics['length']['p95']}/{metrics['length']['max']}\n\n"
        "Natural-stop rows are reused byte-for-byte. Only source cap-hit prompts are regenerated from "
        "their original input and seed, and publication fails on any 16K prefix mismatch.\n",
    )
    print(
        json.dumps({key: value for key, value in metrics.items() if key != "per_problem"}, indent=2),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.command == "shard":
        run_shard(args)
    elif args.command == "aggregate":
        aggregate(args)
    elif args.command == "cap-rerun-shard":
        run_cap_rerun_shard(args)
    else:
        aggregate_cap_proxy(args)


if __name__ == "__main__":
    main()
