#!/usr/bin/env python3
"""Generate one resumable 106-prompt split draw for the AIME split-control eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

TARGET_CONTEXT = 32768
PROMPTS = 106
DRAWS = 4
SEED_BASE = 1234
SEED_SPAN = 480
STOP_TOKENS = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "n": 1}
EVAL_CONTRACT_SHA256 = "e7b4d14ab32512d5d23a0820e032fd5418a8c2d9cbd391af0e3016acab0e01f7"
SPLIT_SHA256 = {
    "held_in": "44f0ed73482564e0f59343c913fad6b6e526bef5287f6216a495cba00b013333",
    "held_out": "4e1b724926f7a50b4ee6e05710c18dd750e9d196dd4596e2aad2c36521e6fefa",
}


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
    """Read a resumable record log, discarding only a torn final line."""
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


def sample_seed(draw_index: int, eval_index: int) -> int:
    """Seed a single completion.

    ``eval_index`` is inherited from the 480-problem AIME 2009-2024 corpus and is
    unique across both 106-problem splits, so this is collision-free without
    renumbering the rows.
    """
    if draw_index not in range(DRAWS):
        raise ValueError(f"draw_index must be 0..{DRAWS - 1}: {draw_index}")
    if eval_index not in range(SEED_SPAN):
        raise ValueError(f"eval_index must be 0..{SEED_SPAN - 1}: {eval_index}")
    return SEED_BASE + SEED_SPAN * draw_index + eval_index


def response_budget(prompt_tokens: int) -> int:
    budget = TARGET_CONTEXT - prompt_tokens
    if prompt_tokens < 1 or budget < 1:
        raise RuntimeError(f"prompt leaves no response budget: {prompt_tokens}")
    return budget


def generation_policy(args: argparse.Namespace, overlay_identity: str) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "checkpoint_identity": args.checkpoint_identity,
        "decoding": DECODING,
        "draw_index": args.draw_index,
        "eval_contract_sha256": EVAL_CONTRACT_SHA256,
        "prompt": {
            "source_field": "rendered_prompt",
            "uses_qwen2_5_default_chat_template": False,
            "verified_against": "rendered_prompt_sha256",
        },
        "response_budget": "32768 - rendered_prompt_tokens",
        "seed_formula": "1234 + 480 * draw_index + eval_index",
        "split": args.split,
        "split_sha256": SPLIT_SHA256[args.split],
        "stop_token_ids": sorted(STOP_TOKENS.values()),
        "target": args.target,
        "target_context": TARGET_CONTEXT,
        "tokenizer_overlay_identity": overlay_identity,
    }


def reconcile(path: Path, policy_sha256: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for row in recover_records(path):
        index = int(row.get("eval_index", -1))
        if index in range(SEED_SPAN) and row.get("policy_sha256") == policy_sha256:
            records[index] = row
    ordered = sorted(records.values(), key=lambda row: int(row["eval_index"]))
    if path.exists() or ordered:
        atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered))
    return records


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    eval_file = Path(args.eval_file).resolve()
    if sha256_file(eval_file) != SPLIT_SHA256[args.split]:
        raise RuntimeError(f"evaluation split hash changed: {eval_file}")
    rows = read_jsonl(eval_file)
    if len(rows) != PROMPTS:
        raise RuntimeError(f"expected {PROMPTS} prompts, found {len(rows)}")
    indices = sorted(int(row.get("eval_index", -1)) for row in rows)
    if len(set(indices)) != PROMPTS or indices[0] < 0 or indices[-1] >= SEED_SPAN:
        raise RuntimeError(f"eval indices are not canonical: {indices[:4]}...")
    for row in rows:
        rendered = row.get("rendered_prompt")
        accepted = row.get("accepted_answer_integers")
        if not isinstance(rendered, str) or sha256_text(rendered) != row.get("rendered_prompt_sha256"):
            raise RuntimeError(f"rendered prompt hash mismatch: {row.get('problem_id')}")
        if row.get("split") != args.split:
            raise RuntimeError(f"row split mismatch: {row.get('problem_id')}")
        if (
            not isinstance(accepted, list)
            or not accepted
            or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 999 for v in accepted)
        ):
            raise RuntimeError(f"invalid accepted answers: {row.get('problem_id')}")

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
    pending = [row for row in rows if int(row["eval_index"]) not in completed]
    if not pending:
        print(
            f"SPLIT_AIME_DRAW_REUSED target={args.target} split={args.split} draw={args.draw_index}",
            flush=True,
        )
        return

    prepared: list[tuple[dict[str, Any], list[int], int]] = []
    for row in pending:
        # The rendered prompt is used verbatim; applying the chat template here
        # would inject a system turn that neither generation nor training saw.
        input_ids = [int(value) for value in tokenizer.encode(row["rendered_prompt"], add_special_tokens=False)]
        prepared.append((row, input_ids, response_budget(len(input_ids))))

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
                    min_p=DECODING["min_p"],
                    max_tokens=item[2],
                    seed=sample_seed(args.draw_index, int(item[0]["eval_index"])),
                    stop_token_ids=sorted(STOP_TOKENS.values()),
                    ignore_eos=False,
                )
                for item in chunk
            ]
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
            if len(outputs) != len(chunk):
                raise RuntimeError("vLLM returned a different number of outputs")
            for (row, input_ids, budget), output in zip(chunk, outputs, strict=True):
                completion = output.outputs[0]
                token_ids = [int(value) for value in (completion.token_ids or [])]
                finish_reason = str(completion.finish_reason or "")
                stop_reason = completion.stop_reason
                if isinstance(stop_reason, int):
                    stop_reason = int(stop_reason)
                elif stop_reason is not None:
                    stop_reason = str(stop_reason)
                response = str(completion.text)
                index = int(row["eval_index"])
                record = {
                    "accepted_answer_integers": [int(value) for value in row["accepted_answer_integers"]],
                    "artifact_schema_version": 1,
                    "cap_hit": finish_reason == "length" or len(token_ids) >= budget,
                    "checkpoint_identity": args.checkpoint_identity,
                    "contest": str(row["contest"]),
                    "draw_index": args.draw_index,
                    "effective_max_response_tokens": budget,
                    "eval_index": index,
                    "finish_reason": finish_reason,
                    "generated_token_count": len(token_ids),
                    "generated_token_ids": token_ids,
                    "policy_sha256": fingerprint,
                    "problem_id": str(row["problem_id"]),
                    "rendered_prompt_sha256": str(row["rendered_prompt_sha256"]),
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
                f"SPLIT_AIME_PROGRESS target={args.target} split={args.split} "
                f"draw={args.draw_index} completed={len(completed)}/{PROMPTS}",
                flush=True,
            )
    finally:
        del llm
    completed = reconcile(records_path, fingerprint)
    if set(completed) != {int(row["eval_index"]) for row in rows}:
        raise RuntimeError(f"generation ended without all {PROMPTS} prompts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint-identity", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--split", required=True, choices=sorted(SPLIT_SHA256))
    parser.add_argument("--draw-index", required=True, type=int)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = parser.parse_args()
    sample_seed(args.draw_index, 0)
    return args


if __name__ == "__main__":
    run(parse_args())
