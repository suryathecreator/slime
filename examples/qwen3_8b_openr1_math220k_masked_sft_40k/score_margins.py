#!/usr/bin/env python3
"""Score wrong OpenR1 traces under ordinary and answer-conditioned prompts."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    atomic_json,
    normalize_token_ids,
    token_sha256,
    user_content,
)


def parse_args() -> argparse.Namespace:
    """Parse a single-GPU scoring shard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-traces", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument("--expected-wrong", type=int, default=20000)
    parser.add_argument(
        "--chat-template-kwargs",
        default='{"enable_thinking": true}',
        help="JSON object forwarded to apply_chat_template",
    )
    return parser.parse_args()


def load_assigned(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load this shard's wrong traces and validate their stable indices."""
    assigned: list[dict[str, Any]] = []
    total_wrong = 0
    with Path(args.selected_traces).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("is_correct"):
                continue
            wrong_index = int(row["wrong_index"])
            if wrong_index != total_wrong:
                raise ValueError(f"noncontiguous wrong index {wrong_index}, expected {total_wrong}")
            if wrong_index % args.num_shards == args.shard_index:
                assigned.append(row)
            total_wrong += 1
    if total_wrong != args.expected_wrong:
        raise ValueError(f"expected {args.expected_wrong} wrong traces, found {total_wrong}")
    expected_assigned = sum(1 for index in range(total_wrong) if index % args.num_shards == args.shard_index)
    if len(assigned) != expected_assigned:
        raise ValueError("assigned shard count mismatch")
    return assigned


def record_path(output_dir: str | Path, wrong_index: int) -> Path:
    """Return the atomic record path for one global wrong-trace index."""
    return Path(output_dir) / "records" / f"wrong_{wrong_index:05d}.json"


def compatible_record(path: Path, row: dict[str, Any]) -> bool:
    """Return whether an existing atomic score is safe to resume."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    margins = record.get("delta_logprobs")
    return (
        record.get("trace_id") == row.get("trace_id")
        and record.get("assistant_token_sha256") == row.get("assistant_token_sha256")
        and isinstance(margins, list)
        and len(margins) == int(row.get("assistant_token_count", -1))
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in margins)
    )


def atomic_record(path: Path, value: dict[str, Any]) -> None:
    """Commit one score record atomically for exact resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def first_parameter_device(model: Any) -> Any:
    """Return the device used by an unsharded Hugging Face model."""
    return next(model.parameters()).device


def score_batch(model: Any, sequences: list[list[int]], prompt_lengths: list[int], trace_ids: list[list[int]]) -> list[list[float]]:
    """Compute causal trace-token log probabilities without materializing FP32 logits."""
    import torch

    max_length = max(len(sequence) for sequence in sequences)
    pad_id = int(model.config.pad_token_id)
    padded = [sequence + [pad_id] * (max_length - len(sequence)) for sequence in sequences]
    attention = [[1] * len(sequence) + [0] * (max_length - len(sequence)) for sequence in sequences]
    device = first_parameter_device(model)
    input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention, dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_tensor, attention_mask=attention_tensor, use_cache=False).logits
    results: list[list[float]] = []
    chunk_size = 256
    for batch_index, (prompt_length, targets) in enumerate(zip(prompt_lengths, trace_ids, strict=True)):
        row_values: list[float] = []
        start = prompt_length - 1
        for offset in range(0, len(targets), chunk_size):
            count = min(chunk_size, len(targets) - offset)
            chunk = logits[batch_index, start + offset : start + offset + count].float()
            target_tensor = torch.tensor(targets[offset : offset + count], dtype=torch.long, device=device)
            selected = chunk.gather(1, target_tensor[:, None]).squeeze(1)
            values = selected - torch.logsumexp(chunk, dim=-1)
            row_values.extend(float(value) for value in values.cpu().tolist())
        results.append(row_values)
    del logits, input_tensor, attention_tensor
    return results


def score_prompt_variant(
    tokenizer: Any,
    model: Any,
    rows: list[dict[str, Any]],
    conditioned: bool,
    max_length: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[list[float]]:
    """Score a batch with or without the known correct answer in the prompt."""
    prompts: list[list[int]] = []
    traces: list[list[int]] = []
    sequences: list[list[int]] = []
    for row in rows:
        answer = str(row["answer"]) if conditioned else None
        prompt_ids = normalize_token_ids(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": user_content(str(row["problem"]), answer),
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                **(chat_template_kwargs or {}),
            )
        )
        trace_ids = [int(token) for token in tokenizer.encode(str(row["assistant_trace"]), add_special_tokens=False)]
        if token_sha256(trace_ids) != row["assistant_token_sha256"]:
            raise ValueError(f"token hash drift for {row['trace_id']}")
        if len(prompt_ids) + len(trace_ids) > max_length:
            raise ValueError(f"scoring sequence exceeds context for {row['trace_id']}")
        prompts.append(prompt_ids)
        traces.append(trace_ids)
        sequences.append(prompt_ids + trace_ids)
    return score_batch(model, sequences, [len(prompt) for prompt in prompts], traces)


def main() -> None:
    """Load one frozen base-model shard and atomically score pending traces."""
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    assigned = load_assigned(args)
    chat_template_kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("--chat-template-kwargs must decode to a JSON object")
    pending = [row for row in assigned if not compatible_record(record_path(args.output_dir, int(row["wrong_index"])), row)]
    print(
        f"MARGIN_SHARD_START shard={args.shard_index}/{args.num_shards} assigned={len(assigned)} pending={len(pending)}",
        flush=True,
    )
    if pending:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=True,
        ).to("cuda")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            unconditioned = score_prompt_variant(
                tokenizer,
                model,
                batch,
                False,
                args.max_sequence_length,
                chat_template_kwargs,
            )
            conditioned = score_prompt_variant(
                tokenizer,
                model,
                batch,
                True,
                args.max_sequence_length,
                chat_template_kwargs,
            )
            for row, uncond, cond in zip(batch, unconditioned, conditioned, strict=True):
                if len(uncond) != len(cond) or len(uncond) != int(row["assistant_token_count"]):
                    raise ValueError(f"score alignment failure for {row['trace_id']}")
                margins = [float(right) - float(left) for left, right in zip(uncond, cond, strict=True)]
                if any(not math.isfinite(value) for value in margins):
                    raise ValueError(f"non-finite margin for {row['trace_id']}")
                record = {
                    "assistant_token_count": int(row["assistant_token_count"]),
                    "assistant_token_sha256": row["assistant_token_sha256"],
                    "delta_logprobs": margins,
                    "margin_definition": "logp_answer_conditioned_minus_logp_unconditioned",
                    "problem_id": row["problem_id"],
                    "shard_index": args.shard_index,
                    "trace_id": row["trace_id"],
                    "wrong_index": int(row["wrong_index"]),
                }
                atomic_record(record_path(args.output_dir, int(row["wrong_index"])), record)
            print(
                f"MARGIN_SHARD_PROGRESS shard={args.shard_index} committed={min(start + len(batch), len(pending))}/{len(pending)}",
                flush=True,
            )

    complete = sum(
        compatible_record(record_path(args.output_dir, int(row["wrong_index"])), row) for row in assigned
    )
    if complete != len(assigned):
        raise RuntimeError(f"shard incomplete: {complete}/{len(assigned)}")
    atomic_json(
        Path(args.output_dir) / f"shard_{args.shard_index:02d}_stats.json",
        {
            "assigned": len(assigned),
            "complete": complete,
            "model": args.model,
            "chat_template_kwargs": chat_template_kwargs,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
    )
    print(f"MARGIN_SHARD_COMPLETE shard={args.shard_index} records={complete}", flush=True)


if __name__ == "__main__":
    main()
