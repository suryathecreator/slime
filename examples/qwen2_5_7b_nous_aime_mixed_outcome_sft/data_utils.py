"""Deterministic data and exact-token helpers for the Nous AIME experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    assert_record,
    atomic_json,
    atomic_jsonl,
    length_summary,
    sha256_file,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    CHAT_BOUNDARY_LITERALS,
    chat_prefix_suffix_ids,
    normalize_token_ids,
    protected_assistant_positions,
    token_sha256,
    tokenizer_control_inventory,
)

VARIANTS = (
    "aime24_correct_3000",
    "aime24_incorrect_3000",
    "aime25_correct_3000",
    "aime25_incorrect_3000",
)
YEARS = ("aime24", "aime25")
DATASET_ROWS = 3000
GLOBAL_BATCH_SIZE = 64
UPDATES = math.ceil(DATASET_ROWS / GLOBAL_BATCH_SIZE)
RUNTIME_PRESENTATIONS = UPDATES * GLOBAL_BATCH_SIZE
WRAP_PRESENTATIONS = RUNTIME_PRESENTATIONS - DATASET_ROWS


def stable_hex(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ordered_values_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    if any(literal in value for literal in CHAT_BOUNDARY_LITERALS):
        raise ValueError(f"{field} contains a Qwen chat boundary")
    return value


def source_trace_id(year: str, doc_id: str, response_id: int) -> str:
    return f"{year}:doc_{doc_id}:response_{response_id:02d}"


def build_record(
    tokenizer: Any,
    *,
    query: str,
    assistant: str,
    metadata: dict[str, Any],
    max_sequence_length: int,
) -> dict[str, Any]:
    """Render exact Nous user/assistant bytes into the weighted-SFT record."""
    query = exact_nonempty_text(query, "query")
    assistant = exact_nonempty_text(assistant, "assistant")
    prefix, suffix = chat_prefix_suffix_ids(tokenizer, query, {})
    assistant_ids = normalize_token_ids(tokenizer.encode(assistant, add_special_tokens=False))
    if not assistant_ids:
        raise ValueError("assistant tokenized to an empty sequence")
    total_tokens = len(prefix) + len(assistant_ids) + len(suffix)
    if total_tokens > max_sequence_length:
        raise ValueError(f"rendered sequence has {total_tokens} tokens, cap={max_sequence_length}")
    protected, reasons, literal_positions = protected_assistant_positions(tokenizer, assistant_ids, assistant)
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    im_start_id = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    eos_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    if (eos_id, im_start_id, im_end_id) != (151643, 151644, 151645):
        raise ValueError("Qwen control-token IDs drifted")
    if suffix != [im_end_id, 198]:
        raise ValueError(f"unexpected Qwen assistant suffix: {suffix}")
    if any(token in assistant_ids for token in (eos_id, im_start_id, im_end_id)):
        raise ValueError("assistant embeds a Qwen chat-boundary token")
    result_metadata = dict(metadata)
    result_metadata.update(
        {
            "assistant_token_count": len(assistant_ids),
            "assistant_token_sha256": token_sha256(assistant_ids),
            "loss_weights": [1.0] * len(assistant_ids) + [1.0, 0.0],
            "prefix_token_count": len(prefix),
            "protected_assistant_positions": protected,
            "protected_literal_positions": literal_positions,
            "protected_token_reasons": reasons,
            "query_sha256": sha256_text(query),
            "response_length": len(assistant_ids) + len(suffix),
            "total_token_count": total_tokens,
            "train_terminal_policy": "assistant_plus_fully_supervised_im_end_no_synthetic_endoftext",
        }
    )
    record = {"input_ids": prefix + assistant_ids + suffix, "metadata": result_metadata}
    assert_record(record, max_sequence_length)
    return record


def conditionally_uniform_subset(
    records: list[dict[str, Any]],
    *,
    count: int,
    required_doc_ids: set[str],
    seed: int,
    year: str,
) -> tuple[list[dict[str, Any]], int]:
    """Choose a keyed-random subset, conditioning only on problem coverage."""
    if count > len(records):
        raise ValueError(f"{year} has only {len(records)} eligible correct rows for target {count}")
    for attempt in range(10000):
        ordered = sorted(
            records,
            key=lambda row: (
                stable_hex(
                    "nous-aime-correct-subset-v1",
                    seed,
                    year,
                    attempt,
                    row["metadata"]["base_trace_id"],
                ),
                row["metadata"]["base_trace_id"],
            ),
        )
        selected = ordered[:count]
        observed = {str(row["metadata"]["doc_id"]) for row in selected}
        if observed == required_doc_ids:
            return selected, attempt
    raise RuntimeError(f"failed to sample a covered {year} correct subset after 10000 attempts")


def expand_balanced(
    records: list[dict[str, Any]], *, variant: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize 3K rows and make the eight runtime wrap exposures balanced."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    base_ids = [str(row["metadata"]["base_trace_id"]) for row in records]
    if len(base_ids) != len(set(base_ids)) or not base_ids:
        raise ValueError(f"{variant} source trace IDs must be unique and nonempty")
    base_by_id = dict(zip(base_ids, records))
    copies, extra_count = divmod(DATASET_ROWS, len(records))
    extra_ids = set(
        sorted(
            base_ids,
            key=lambda trace_id: (
                stable_hex("nous-aime-physical-extra-v1", seed, variant, trace_id),
                trace_id,
            ),
        )[:extra_count]
    )
    low_copy_ids = sorted(
        set(base_ids) - extra_ids,
        key=lambda trace_id: (
            stable_hex("nous-aime-runtime-wrap-v1", seed, variant, trace_id),
            trace_id,
        ),
    )
    if len(low_copy_ids) < WRAP_PRESENTATIONS:
        raise ValueError(f"{variant} cannot balance {WRAP_PRESENTATIONS} wrap rows across lower-copy traces")
    wrap_ids = low_copy_ids[:WRAP_PRESENTATIONS]
    instances_by_base: dict[str, list[dict[str, Any]]] = {}
    for base_id in base_ids:
        count = copies + int(base_id in extra_ids)
        instances = []
        for copy_index in range(count):
            record = copy.deepcopy(base_by_id[base_id])
            record["metadata"].update(
                {
                    "copy_index": copy_index,
                    "extra_physical_copy": copy_index == copies,
                    "trace_id": f"{base_id}:copy_{copy_index:02d}",
                    "variant": variant,
                }
            )
            instances.append(record)
        instances_by_base[base_id] = instances
    reserved = [instances_by_base[base_id].pop() for base_id in wrap_ids]
    remaining = [record for values in instances_by_base.values() for record in values]
    remaining.sort(
        key=lambda row: (
            stable_hex("nous-aime-expanded-order-v1", seed, variant, row["metadata"]["trace_id"]),
            row["metadata"]["trace_id"],
        )
    )
    wrap_positions = list(range(DATASET_ROWS))
    random.Random(seed + 1).shuffle(wrap_positions)
    wrap_positions = wrap_positions[:WRAP_PRESENTATIONS]
    expanded: list[dict[str, Any] | None] = [None] * DATASET_ROWS
    for position, record in zip(wrap_positions, reserved):
        expanded[position] = record
    remaining_iter = iter(remaining)
    for position in range(DATASET_ROWS):
        if expanded[position] is None:
            expanded[position] = next(remaining_iter)
    try:
        next(remaining_iter)
    except StopIteration:
        pass
    else:
        raise AssertionError("expanded dataset fill left unused records")
    result = [record for record in expanded if record is not None]
    if len(result) != DATASET_ROWS:
        raise AssertionError(f"{variant} expansion did not produce {DATASET_ROWS} rows")
    physical = Counter(str(row["metadata"]["base_trace_id"]) for row in result)
    runtime = Counter(physical)
    epoch_one = list(range(DATASET_ROWS))
    random.Random(seed + 1).shuffle(epoch_one)
    for index in epoch_one[:WRAP_PRESENTATIONS]:
        runtime[str(result[index]["metadata"]["base_trace_id"])] += 1
    if max(runtime.values()) - min(runtime.values()) != 1:
        raise ValueError(f"{variant} runtime source exposure differs by more than one")
    return result, {
        "base_trace_count": len(base_ids),
        "dataset_rows": len(result),
        "effective_runtime_presentations": RUNTIME_PRESENTATIONS,
        "extra_physical_trace_ids_sha256": ordered_values_sha256(sorted(extra_ids)),
        "physical_exposure_histogram": {str(key): value for key, value in sorted(Counter(physical.values()).items())},
        "runtime_exposure_histogram": {str(key): value for key, value in sorted(Counter(runtime.values()).items())},
        "wrap_source_trace_ids_sha256": ordered_values_sha256(wrap_ids),
    }


__all__ = [
    "DATASET_ROWS",
    "GLOBAL_BATCH_SIZE",
    "RUNTIME_PRESENTATIONS",
    "UPDATES",
    "VARIANTS",
    "WRAP_PRESENTATIONS",
    "YEARS",
    "assert_record",
    "atomic_json",
    "atomic_jsonl",
    "build_record",
    "conditionally_uniform_subset",
    "expand_balanced",
    "length_summary",
    "normalize_token_ids",
    "ordered_values_sha256",
    "read_jsonl",
    "sha256_file",
    "sha256_text",
    "source_trace_id",
    "stable_hex",
    "tokenizer_control_inventory",
]
