"""Deterministic selection and exact-token helpers for the AIME experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    CHAT_BOUNDARY_LITERALS,
    THINK_LITERALS,
    atomic_json,
    atomic_jsonl,
    chat_prefix_suffix_ids,
    normalize_token_ids,
    protected_assistant_positions,
    sha256_file,
    token_sha256,
    tokenizer_control_inventory,
    trace_set_sha256,
)

IMAGE_PROBLEM_IDS = frozenset(
    {
        "1983-12",
        "1985-15",
        "1985-4",
        "1985-6",
        "1987-15",
        "1987-9",
        "1990-14",
        "1992-12",
        "1993-9",
        "1995-1",
        "1995-9",
        "1999-4",
        "2002-II-4",
        "2007-I-10",
    }
)
MASK_VARIANTS = tuple(f"continue_random_mask_{rate}" for rate in range(10, 100, 10))
CHILD_VARIANTS = (
    "continue_wrong_unmasked",
    *MASK_VARIANTS,
    "continue_correct_only_1500",
)
ALL_TRAINED_VARIANTS = ("aime_correct_3000", *CHILD_VARIANTS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_hex(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_unit_interval(seed: int, trace_id: str, token_index: int) -> float:
    digest = hashlib.sha256(f"{seed}\0{trace_id}\0{token_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def ordered_values_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(str(value) for value in values) + "\n").encode("utf-8")).hexdigest()


def validate_plain_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    if any(literal in value for literal in CHAT_BOUNDARY_LITERALS):
        raise ValueError(f"{field} embeds a Qwen chat boundary")
    return value


def aime_assistant_text(reasoning_content: Any, content: Any) -> str:
    reasoning = validate_plain_text(reasoning_content, "reasoning_content").strip()
    answer = validate_plain_text(content, "content").strip()
    if any(literal in reasoning or literal in answer for literal in THINK_LITERALS):
        raise ValueError("AIME fields already contain think markup")
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def build_prompt_record(
    tokenizer: Any,
    *,
    user_prompt: str,
    assistant_text: str,
    metadata: dict[str, Any],
    max_sequence_length: int,
) -> dict[str, Any]:
    """Render a verbatim user prompt into the exact weighted-SFT record shape."""
    user_prompt = validate_plain_text(user_prompt, "prompt")
    if any(literal in assistant_text for literal in CHAT_BOUNDARY_LITERALS):
        raise ValueError("assistant text embeds a Qwen chat boundary")
    prefix, suffix = chat_prefix_suffix_ids(tokenizer, user_prompt, {})
    assistant_ids = normalize_token_ids(tokenizer.encode(assistant_text, add_special_tokens=False))
    if not assistant_ids:
        raise ValueError("assistant text tokenized to an empty sequence")
    total_tokens = len(prefix) + len(assistant_ids) + len(suffix)
    if total_tokens > max_sequence_length:
        raise ValueError(f"rendered sequence has {total_tokens} tokens, cap={max_sequence_length}")
    protected, reasons, literal_positions = protected_assistant_positions(tokenizer, assistant_ids, assistant_text)
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    im_start_id = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    eos_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    if (eos_id, im_start_id, im_end_id) != (151643, 151644, 151645):
        raise ValueError("Qwen control-token IDs drifted")
    if any(token in assistant_ids for token in (eos_id, im_start_id, im_end_id)):
        raise ValueError("assistant token IDs embed a Qwen chat boundary")
    if suffix != [im_end_id, 198]:
        raise ValueError(f"unexpected Qwen assistant suffix: {suffix}")
    response_weights = [1.0] * len(assistant_ids) + [1.0, 0.0]
    result_metadata = dict(metadata)
    result_metadata.update(
        {
            "assistant_token_count": len(assistant_ids),
            "assistant_token_sha256": token_sha256(assistant_ids),
            "loss_weights": response_weights,
            "prefix_token_count": len(prefix),
            "protected_assistant_positions": protected,
            "protected_literal_positions": literal_positions,
            "protected_token_reasons": reasons,
            "response_length": len(assistant_ids) + len(suffix),
            "total_token_count": total_tokens,
            "train_terminal_policy": ("assistant_plus_fully_supervised_im_end_no_synthetic_endoftext"),
        }
    )
    return {
        "input_ids": prefix + assistant_ids + suffix,
        "metadata": result_metadata,
    }


def length_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty record set")

    def summary(values: list[int]) -> dict[str, float | int]:
        ordered = sorted(values)

        def percentile(probability: float) -> int:
            index = max(0, math.ceil(probability * len(ordered)) - 1)
            return ordered[index]

        return {
            "mean": sum(ordered) / len(ordered),
            "median": percentile(0.5),
            "p90": percentile(0.9),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "min": ordered[0],
            "max": ordered[-1],
        }

    return {
        "rows": len(records),
        "prefix_tokens": summary([int(row["metadata"]["prefix_token_count"]) for row in records]),
        "assistant_tokens": summary([int(row["metadata"]["assistant_token_count"]) for row in records]),
        "response_tokens": summary([int(row["metadata"]["response_length"]) for row in records]),
        "total_tokens": summary([len(row["input_ids"]) for row in records]),
    }


def assert_record(record: dict[str, Any], max_sequence_length: int) -> None:
    ids = record.get("input_ids")
    metadata = record.get("metadata")
    if not isinstance(ids, list) or not ids or not isinstance(metadata, dict):
        raise ValueError("malformed pretokenized record")
    ids = [int(value) for value in ids]
    response_length = int(metadata["response_length"])
    assistant_count = int(metadata["assistant_token_count"])
    weights = [float(value) for value in metadata["loss_weights"]]
    protected = [int(value) for value in metadata["protected_assistant_positions"]]
    if len(ids) > max_sequence_length:
        raise ValueError("record exceeds the total sequence cap")
    if len(weights) != response_length or response_length != assistant_count + 2:
        raise ValueError("response/weight boundary mismatch")
    if ids[-2:] != [151645, 198] or weights[-2:] != [1.0, 0.0]:
        raise ValueError("terminal supervision policy drift")
    if 151643 in ids:
        raise ValueError("synthetic endoftext is forbidden")
    if not protected or any(index < 0 or index >= assistant_count or weights[index] != 1.0 for index in protected):
        raise ValueError("protected assistant supervision drift")
    if any(not math.isfinite(value) or value not in (0.0, 1.0) for value in weights):
        raise ValueError("loss weights must be binary and finite")


__all__ = [
    "ALL_TRAINED_VARIANTS",
    "CHILD_VARIANTS",
    "IMAGE_PROBLEM_IDS",
    "MASK_VARIANTS",
    "aime_assistant_text",
    "assert_record",
    "atomic_json",
    "atomic_jsonl",
    "build_prompt_record",
    "length_summary",
    "normalize_token_ids",
    "ordered_values_sha256",
    "read_jsonl",
    "sha256_file",
    "stable_hex",
    "stable_unit_interval",
    "token_sha256",
    "tokenizer_control_inventory",
    "trace_set_sha256",
]
