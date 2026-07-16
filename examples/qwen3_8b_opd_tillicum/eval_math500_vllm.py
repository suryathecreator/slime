#!/usr/bin/env python3
"""Run or merge a vLLM MATH-500 eval shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from numbers import Integral
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
    shard.add_argument("--kv-cache-dtype", default="auto")
    shard.add_argument("--safetensors-load-strategy", default="prefetch")
    shard.add_argument("--enable-prefix-caching", action="store_true")
    shard.add_argument("--enable-chunked-prefill", action="store_true")
    shard.add_argument("--async-scheduling", action="store_true")
    shard.add_argument("--speculative-method", default="none")
    shard.add_argument("--num-speculative-tokens", type=int, default=0)
    shard.add_argument("--prompt-lookup-min", type=int, default=5)
    shard.add_argument("--prompt-lookup-max", type=int, default=5)
    shard.add_argument("--trust-remote-code", action="store_true")
    shard.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    shard.add_argument("--chunk-size", type=int, default=16)
    shard.add_argument("--resume-completed", action="store_true")
    shard.add_argument("--enable-thinking", action="store_true")
    shard.add_argument("--stop-token-ids", type=int, nargs="+", required=True)
    shard.add_argument("--im-end-token-id", type=int, required=True)
    shard.add_argument("--endoftext-token-id", type=int, required=True)
    shard.add_argument("--preserve-special-tokens", action="store_true")

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


ARTIFACT_SCHEMA_VERSION = 2


def build_prompt(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def generation_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": str(Path(args.model).resolve()),
        "enable_thinking": bool(args.enable_thinking),
        "stop_token_ids": sorted(set(args.stop_token_ids)),
        "im_end_token_id": args.im_end_token_id,
        "endoftext_token_id": args.endoftext_token_id,
        "preserve_special_tokens": bool(args.preserve_special_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_response_len": args.max_response_len,
        "max_context_len": args.max_context_len,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
    }


def generation_policy_fingerprint(policy: dict[str, Any]) -> str:
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_qwen3_control_tokens(tokenizer: Any, args: argparse.Namespace) -> None:
    expected = {
        "<|im_end|>": args.im_end_token_id,
        "<|endoftext|>": args.endoftext_token_id,
    }
    for token, expected_id in expected.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise RuntimeError(f"Qwen3 token id mismatch for {token}: expected {expected_id}, found {actual_id}")
    missing_stops = set(expected.values()) - set(args.stop_token_ids)
    if missing_stops:
        raise RuntimeError(f"Qwen3 stop-token list is missing required ids: {sorted(missing_stops)}")


def chunk_path(stage_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> Path:
    first = rows[0]["_eval_index"]
    last = rows[-1]["_eval_index"]
    return stage_dir / f"debug_eval_chunk_shard{shard_index}_start{first}_end{last}.pt"


def sample_path(stage_dir: Path, shard_index: int, eval_index: int) -> Path:
    return stage_dir / f"debug_eval_chunk_shard{shard_index}_start{eval_index}_end{eval_index}.pt"


def artifact_paths(stage_dir: Path) -> list[Path]:
    paths = set(stage_dir.glob("debug_eval_shard_*.pt"))
    paths.update(stage_dir.glob("debug_eval_chunk_shard*_start*_end*.pt"))
    return sorted(paths)


def load_compatible_artifact(
    path: Path,
    expected_policy: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Ignoring unreadable vLLM artifact {path}: {exc}")
        return None
    found_policy = payload.get("generation_policy")
    expected_fingerprint = generation_policy_fingerprint(expected_policy)
    found_fingerprint = payload.get("generation_policy_fingerprint")
    if found_policy != expected_policy or found_fingerprint != expected_fingerprint:
        print(
            f"Ignoring incompatible vLLM artifact {path}: "
            f"expected_policy={expected_fingerprint} "
            f"found_policy={generation_policy_fingerprint(found_policy) if isinstance(found_policy, dict) else '<missing>'} "
            f"found_fingerprint={found_fingerprint or '<missing>'}"
        )
        return None
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        print(f"Ignoring empty vLLM artifact {path}")
        return None
    return samples


def discover_completed_samples(
    stage_dir: Path,
    expected_policy: dict[str, Any],
    allowed_indices: set[int],
) -> dict[int, tuple[Path, dict[str, Any]]]:
    completed: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in artifact_paths(stage_dir):
        samples = load_compatible_artifact(path, expected_policy)
        if samples is None:
            continue
        artifact_indices: set[int] = set()
        for sample in samples:
            eval_index = int(sample["eval_index"])
            if eval_index not in allowed_indices:
                raise RuntimeError(f"Unexpected eval_index={eval_index} in {path}")
            if eval_index in artifact_indices:
                raise RuntimeError(f"Duplicate eval_index={eval_index} inside {path}")
            artifact_indices.add(eval_index)
            if eval_index in completed:
                raise RuntimeError(
                    f"Duplicate eval_index={eval_index} in {path}; already found in {completed[eval_index][0]}"
                )
            completed[eval_index] = (path, sample)
    return completed


def load_valid_chunk(
    path: Path,
    expected_indices: set[int],
    expected_policy: dict[str, Any],
) -> list[dict[str, Any]] | None:
    samples = load_compatible_artifact(path, expected_policy)
    if samples is None:
        return None
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
) -> tuple[list[tuple[int, list[int], dict[str, Any]]], list[dict[str, Any]]]:
    prepared = []
    zero_cap_samples = []
    for row in rows:
        rendered_prompt = build_prompt(tokenizer, str(row["prompt"]), enable_thinking=args.enable_thinking)
        prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
        available = args.max_context_len - len(prompt_ids)
        effective_cap = min(args.max_response_len, max(available, 0))
        base = {
            "eval_index": row["_eval_index"],
            "prompt": row["prompt"],
            "label": row["label"],
            "prefill_text": rendered_prompt,
            "prefill_token_ids": list(prompt_ids),
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
            zero_cap_samples.append(
                {
                    **base,
                    "generated_token_ids": [],
                    "terminal_stop_token_id": None,
                    "generated_token_ids_with_terminal": [],
                    "response": "",
                    "response_plain": "",
                    "response_length": 0,
                    "status": "truncated",
                    "finish_reason": "length",
                }
            )
        else:
            prepared.append((effective_cap, list(prompt_ids), base))
    return prepared, zero_cap_samples


def sample_from_output(
    tokenizer: Any,
    base: dict[str, Any],
    cap: int,
    output: Any,
) -> dict[str, Any]:
    completion = output.outputs[0]
    token_ids = [int(token_id) for token_id in (getattr(completion, "token_ids", []) or [])]
    finish_reason = str(getattr(completion, "finish_reason", ""))
    stop_reason = getattr(completion, "stop_reason", None)
    terminal_stop_token_id = (
        int(stop_reason)
        if isinstance(stop_reason, Integral) or (isinstance(stop_reason, str) and stop_reason.isdigit())
        else None
    )
    token_ids_with_terminal = list(token_ids)
    if terminal_stop_token_id is not None and (
        not token_ids_with_terminal or token_ids_with_terminal[-1] != terminal_stop_token_id
    ):
        token_ids_with_terminal.append(terminal_stop_token_id)
    response = tokenizer.decode(
        token_ids_with_terminal,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    response_plain = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    status = "truncated" if finish_reason == "length" or len(token_ids) >= cap else "completed"
    return {
        **base,
        "generated_token_ids": token_ids,
        "terminal_stop_token_id": terminal_stop_token_id,
        "generated_token_ids_with_terminal": token_ids_with_terminal,
        "response": response,
        "response_plain": response_plain,
        "vllm_response_text": tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "response_length": len(token_ids),
        "status": status,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
    }


def speculative_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.speculative_method.lower() in {"", "none", "off", "disabled"}:
        return None
    if args.num_speculative_tokens <= 0:
        raise ValueError("Speculative decoding requires --num-speculative-tokens > 0")
    return {
        "method": args.speculative_method,
        "num_speculative_tokens": args.num_speculative_tokens,
        "prompt_lookup_min": args.prompt_lookup_min,
        "prompt_lookup_max": args.prompt_lookup_max,
    }


def operational_config(args: argparse.Namespace, vllm_version: str) -> dict[str, Any]:
    return {
        "vllm_version": vllm_version,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_dtype": args.kv_cache_dtype,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "offline_llm": True,
        "async_scheduling": False,
        "safetensors_load_strategy": args.safetensors_load_strategy,
        "speculative_config": None,
        "skip_tokenizer_init": True,
        "detokenize": False,
        "output_kind": "final_only",
        "persistence": "single_generate_call_then_per_sample_atomic",
    }


def run_offline_batch(
    tokenizer: Any,
    pending_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    on_sample: Any,
) -> None:
    """Load one offline engine and submit the entire shard in one call."""
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    prepared, zero_cap_samples = prepare_chunk_rows(tokenizer, pending_rows, args)
    for sample in zero_cap_samples:
        on_sample(sample)
    if not prepared:
        return
    if speculative_config(args) is not None:
        raise ValueError("Offline evaluation does not permit speculative decoding")
    print(
        "VLLM_EVAL_OPERATIONAL_CONFIG "
        + json.dumps(operational_config(args, vllm.__version__), sort_keys=True),
        flush=True,
    )
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        skip_tokenizer_init=True,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=1,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        safetensors_load_strategy=args.safetensors_load_strategy,
        disable_log_stats=False,
    )
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids) for _, prompt_ids, _ in prepared]
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=cap,
            stop_token_ids=args.stop_token_ids,
            detokenize=False,
            skip_special_tokens=not args.preserve_special_tokens,
            spaces_between_special_tokens=False,
            include_stop_str_in_output=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        for cap, prompt_ids, base in prepared
    ]
    outputs = engine.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    if len(outputs) != len(prepared):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prepared)} prompts")
    for (cap, _prompt_ids, base), output in zip(prepared, outputs, strict=True):
        on_sample(sample_from_output(tokenizer, base, cap, output))
    if hasattr(engine, "shutdown"):
        engine.shutdown()


def run_shard(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}")
    if not args.enable_thinking:
        raise ValueError("Qwen3 MATH-500 eval requires explicit --enable-thinking")
    if not args.preserve_special_tokens:
        raise ValueError("Auditable Qwen3 eval requires --preserve-special-tokens")

    policy = generation_policy(args)
    policy_fingerprint = generation_policy_fingerprint(policy)
    print(f"VLLM_EVAL_GENERATION_POLICY fingerprint={policy_fingerprint} policy={json.dumps(policy, sort_keys=True)}")

    data_path = Path(args.data)
    all_rows = load_rows(data_path)
    all_indices = {int(row["_eval_index"]) for row in all_rows}
    rows = [row for row in all_rows if row["_eval_index"] % args.num_shards == args.shard_index]
    stage_dir = Path(args.out).parent
    completed = discover_completed_samples(stage_dir, policy, all_indices)
    if completed and not args.resume_completed:
        raise RuntimeError(
            f"Found {len(completed)} compatible completed samples in {stage_dir}; "
            "use --resume-completed or an empty output directory"
        )
    pending_rows = [row for row in rows if int(row["_eval_index"]) not in completed]
    shard_completed = len(rows) - len(pending_rows)
    print(
        f"VLLM_EVAL_RESUME shard={args.shard_index}/{args.num_shards} "
        f"completed={shard_completed} pending={len(pending_rows)} global_completed={len(completed)}",
        flush=True,
    )

    if not pending_rows:
        print(f"VLLM_EVAL_SHARD_COMPLETE shard={args.shard_index} samples={len(rows)}")
        return

    from transformers import AutoTokenizer
    from slime.backends.vllm_utils.native_sampler import maybe_force_native_sampler

    if maybe_force_native_sampler():
        print("VLLM_EVAL_FORCE_NATIVE_SAMPLER=1: patched vLLM V1 sampler in eval parent")
    import vllm

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_home, trust_remote_code=True)
    validate_qwen3_control_tokens(tokenizer, args)
    runtime = operational_config(args, vllm.__version__)

    def persist_sample(sample: dict[str, Any]) -> None:
        eval_index = int(sample["eval_index"])
        path = sample_path(stage_dir, args.shard_index, eval_index)
        payload = {
            "samples": [sample],
            "stage": args.stage,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "chunk_start_eval_index": eval_index,
            "chunk_end_eval_index": eval_index,
            "generation_policy": policy,
            "generation_policy_fingerprint": policy_fingerprint,
            "operational_config": runtime,
        }
        save_chunk_atomic(path, payload)
        print(
            f"VLLM_EVAL_SAMPLE_COMMIT shard={args.shard_index}/{args.num_shards} "
            f"eval_index={eval_index} path={path}",
            flush=True,
        )

    run_offline_batch(tokenizer, pending_rows, args, persist_sample)


def merge_shards(args: argparse.Namespace) -> None:
    import torch

    stage_dir = Path(args.stage_dir)
    by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    policy_fingerprint: str | None = None
    merged_policy: dict[str, Any] | None = None

    def add_payload(path: Path, payload: dict[str, Any]) -> None:
        nonlocal merged_policy, policy_fingerprint
        policy = payload.get("generation_policy")
        fingerprint = payload.get("generation_policy_fingerprint")
        if not isinstance(policy, dict) or not isinstance(fingerprint, str):
            print(f"Ignoring legacy eval artifact without generation policy: {path}")
            return
        if fingerprint != generation_policy_fingerprint(policy):
            raise RuntimeError(f"Generation policy fingerprint mismatch inside {path}")
        if policy_fingerprint is None:
            policy_fingerprint = fingerprint
            merged_policy = policy
        elif fingerprint != policy_fingerprint:
            raise RuntimeError(
                f"Mixed generation policies in {stage_dir}: expected {policy_fingerprint}, found {fingerprint} in {path}"
            )
        for sample in payload.get("samples", []):
            eval_index = int(sample["eval_index"])
            if eval_index in by_index:
                raise RuntimeError(
                    f"Duplicate eval_index={eval_index} in {path}; already found in {by_index[eval_index][0]}"
                )
            by_index[eval_index] = (path, sample)

    for path in sorted(stage_dir.glob("debug_eval_shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        add_payload(path, payload)
    for path in sorted(stage_dir.glob("debug_eval_chunk_shard*_start*_end*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        add_payload(path, payload)

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
    torch.save(
        {
            "samples": samples,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "generation_policy": merged_policy,
            "generation_policy_fingerprint": policy_fingerprint,
        },
        out,
    )
    print(f"Wrote merged vLLM debug file: {out} samples={len(samples)} policy={policy_fingerprint}")


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
