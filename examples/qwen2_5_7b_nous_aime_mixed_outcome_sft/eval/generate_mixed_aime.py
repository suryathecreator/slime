#!/usr/bin/env python3
"""Generate one resumable 60-prompt draw for the mixed-outcome AIME eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

TARGET_CONTEXT = 32768
PROMPTS = 60
DRAWS = 16
STOP_TOKENS = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
DECODING = {"temperature": 0.7, "top_p": 0.8, "top_k": -1, "n": 1}
CONTRACT_SHA256 = "f49e6b6337ed2b2186b0412cd874583e79d96d89c8b80a91b13d37f0a6ac8b92"
DATASETS_SHA256 = (
    "0c838ee6351d9636a7fdb62dbf9a1123c3fbf1cd13e61325aa892ac4343f6538",
    "195dd659d4c20a3900875cfe7e83acc9a69c16a7c73ad6d6eb5f61c9fd30f70e",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def recover_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    final_nonempty = nonempty[-1] if nonempty else None
    result: list[dict[str, Any]] = []
    offset = 0
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index == final_nonempty and not line.endswith((b"\n", b"\r")):
                atomic_text(path, payload[:offset].decode("utf-8"))
                break
            raise RuntimeError(f"invalid JSONL at {path}:{index + 1}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object JSONL row at {path}:{index + 1}")
        result.append(value)
        offset += len(line)
    return result


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sample_seed(draw_index: int, global_eval_index: int) -> int:
    if draw_index not in range(DRAWS):
        raise ValueError(f"draw_index must be 0..{DRAWS - 1}: {draw_index}")
    if global_eval_index not in range(PROMPTS):
        raise ValueError(f"global_eval_index must be 0..{PROMPTS - 1}: {global_eval_index}")
    return 1234 + PROMPTS * draw_index + global_eval_index


def response_budget(prompt_tokens: int) -> int:
    budget = TARGET_CONTEXT - prompt_tokens
    if prompt_tokens < 1 or budget < 1:
        raise RuntimeError(f"prompt leaves no response budget: {prompt_tokens}")
    return budget


def generation_policy(args: argparse.Namespace, overlay_identity: str) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "checkpoint_identity": args.checkpoint_identity,
        "contract_sha256": CONTRACT_SHA256,
        "datasets_sha256": list(DATASETS_SHA256),
        "decoding": DECODING,
        "draw_index": args.draw_index,
        "prompt": {
            "add_generation_prompt": True,
            "input_key": "query",
            "messages": "one user message containing query verbatim",
            "system_inserted_by_template": "You are a helpful assistant.",
        },
        "response_budget": "32768 - rendered_prompt_tokens",
        "seed_formula": "1234 + 60 * draw_index + global_eval_index",
        "stop_token_ids": sorted(STOP_TOKENS.values()),
        "target": args.target,
        "target_context": TARGET_CONTEXT,
        "tokenizer_overlay_identity": overlay_identity,
    }


def reconcile(path: Path, policy_sha256: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for row in recover_records(path):
        index = int(row.get("global_eval_index", -1))
        if index in range(PROMPTS) and row.get("policy_sha256") == policy_sha256:
            records[index] = row
    ordered = [records[index] for index in range(PROMPTS) if index in records]
    if path.exists() or ordered:
        atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered))
    return records


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    eval_files = [Path(value).resolve() for value in args.eval_file]
    contract_file = Path(args.contract_file).resolve()
    if sha256_file(contract_file) != CONTRACT_SHA256:
        raise RuntimeError(f"evaluation contract hash mismatch: {contract_file}")
    actual_dataset_hashes = tuple(sha256_file(path) for path in eval_files)
    if actual_dataset_hashes != DATASETS_SHA256:
        raise RuntimeError(f"evaluation dataset hashes changed: {actual_dataset_hashes}")
    rows = [row for path in eval_files for row in read_jsonl(path)]
    if len(rows) != PROMPTS:
        raise RuntimeError(f"expected {PROMPTS} prompts, found {len(rows)}")
    indices = [int(row.get("global_eval_index", -1)) for row in rows]
    if indices != list(range(PROMPTS)):
        raise RuntimeError(f"global eval indices are not canonical: {indices}")
    for row in rows:
        query = row.get("query")
        answer = row.get("answer")
        if not isinstance(query, str) or sha256_text(query) != row.get("query_sha256"):
            raise RuntimeError(f"query hash mismatch: {row.get('problem_id')}")
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer <= 999:
            raise RuntimeError(f"invalid answer: {row.get('problem_id')}")

    tokenizer_path = Path(args.tokenizer).resolve()
    overlay = json.loads((tokenizer_path / "overlay_metadata.json").read_text(encoding="utf-8"))
    if overlay.get("checkpoint_identity") != args.checkpoint_identity:
        raise RuntimeError("tokenizer overlay/checkpoint identity mismatch")
    overlay_identity = str(overlay.get("overlay_identity", ""))
    if len(overlay_identity) != 64:
        raise RuntimeError("invalid tokenizer overlay identity")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    actual_stops = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in STOP_TOKENS}
    if actual_stops != STOP_TOKENS or int(tokenizer.eos_token_id) != STOP_TOKENS["<|endoftext|>"]:
        raise RuntimeError(f"tokenizer stop contract changed: {actual_stops}/{tokenizer.eos_token_id}")

    policy = generation_policy(args, overlay_identity)
    fingerprint = sha256_text(json.dumps(policy, sort_keys=True, separators=(",", ":")))
    output_dir = Path(args.output_dir).resolve()
    policy_path = output_dir / "generation_policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text(encoding="utf-8")) != policy:
        raise RuntimeError(f"refusing incompatible resume policy: {policy_path}")
    atomic_text(policy_path, canonical_json(policy))
    records_path = output_dir / "records.jsonl"
    completed = reconcile(records_path, fingerprint)
    pending = [row for row in rows if int(row["global_eval_index"]) not in completed]
    if not pending:
        print(f"MIXED_AIME_DRAW_REUSED target={args.target} draw={args.draw_index}", flush=True)
        return

    prepared: list[tuple[dict[str, Any], list[int], str, int]] = []
    for row in pending:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["query"]}], tokenize=False, add_generation_prompt=True
        )
        input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
        prepared.append((row, input_ids, rendered, response_budget(len(input_ids))))

    llm = LLM(
        model=args.model,
        tokenizer=str(tokenizer_path),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=TARGET_CONTEXT,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=0,
        enable_prefix_caching=True,
    )
    try:
        for start in range(0, len(prepared), args.chunk_size):
            chunk = prepared[start : start + args.chunk_size]
            prompts = [TokensPrompt(prompt_token_ids=item[1]) for item in chunk]
            sampling = [
                SamplingParams(
                    n=1,
                    temperature=DECODING["temperature"],
                    top_p=DECODING["top_p"],
                    top_k=DECODING["top_k"],
                    max_tokens=item[3],
                    seed=sample_seed(args.draw_index, int(item[0]["global_eval_index"])),
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
                response = str(completion.text)
                index = int(row["global_eval_index"])
                record = {
                    "answer": int(row["answer"]),
                    "artifact_schema_version": 1,
                    "cap_hit": finish_reason == "length" or len(token_ids) >= budget,
                    "checkpoint_identity": args.checkpoint_identity,
                    "doc_id": str(row["doc_id"]),
                    "draw_index": args.draw_index,
                    "effective_max_response_tokens": budget,
                    "finish_reason": finish_reason,
                    "generated_token_count": len(token_ids),
                    "generated_token_ids": token_ids,
                    "global_eval_index": index,
                    "policy_sha256": fingerprint,
                    "problem_id": str(row["problem_id"]),
                    "query_sha256": str(row["query_sha256"]),
                    "rendered_prompt_sha256": sha256_text(rendered),
                    "rendered_prompt_tokens": len(input_ids),
                    "response": response,
                    "response_sha256": sha256_text(response),
                    "seed": sample_seed(args.draw_index, index),
                    "split": str(row["split"]),
                    "stop_reason": stop_reason,
                    "target": args.target,
                    "tokenizer_overlay_identity": overlay_identity,
                    "total_context_limit": TARGET_CONTEXT,
                    "year": int(row["year"]),
                }
                append_jsonl(records_path, record)
                completed[index] = record
            print(
                f"MIXED_AIME_PROGRESS target={args.target} draw={args.draw_index} "
                f"completed={len(completed)}/{PROMPTS}",
                flush=True,
            )
    finally:
        del llm
    completed = reconcile(records_path, fingerprint)
    if set(completed) != set(range(PROMPTS)):
        raise RuntimeError("generation ended without all 60 prompts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint-identity", required=True)
    parser.add_argument("--contract-file", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draw-index", required=True, type=int)
    parser.add_argument("--eval-file", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    sample_seed(args.draw_index, 0)
    if len(args.eval_file) != 2:
        parser.error("exactly two --eval-file arguments are required")
    return args


if __name__ == "__main__":
    run(parse_args())
