"""Deterministic selection and exact no-system SFT rendering helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
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
    normalize_token_ids,
    protected_assistant_positions,
    token_sha256,
    tokenizer_control_inventory,
)

VARIANTS = (
    "held_in_correct_6000",
    "held_in_incorrect_6000",
    "held_out_correct_6000",
    "held_out_incorrect_6000",
)
SPLITS = ("held_in", "held_out")
DATASET_ROWS = 6000
EPOCHS = 2
GLOBAL_BATCH_SIZE = 64
UPDATES = math.ceil(DATASET_ROWS * EPOCHS / GLOBAL_BATCH_SIZE)
RUNTIME_PRESENTATIONS = UPDATES * GLOBAL_BATCH_SIZE
WRAP_PRESENTATIONS = RUNTIME_PRESENTATIONS - DATASET_ROWS * EPOCHS
MIXED_PROBLEM_COUNT = 212
SPLIT_PROBLEM_COUNT = MIXED_PROBLEM_COUNT // 2
SOURCE_TRACE_COUNT = 508
PROMPT_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n{problem}"


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


def rendered_user_prefix(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def build_record(
    tokenizer: Any,
    *,
    prompt: str,
    rendered_prompt: str,
    response: str,
    metadata: dict[str, Any],
    max_sequence_length: int,
) -> dict[str, Any]:
    """Build the literal one-user/no-system Qwen training sequence."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be nonempty text")
    if any(literal in prompt or literal in response for literal in CHAT_BOUNDARY_LITERALS):
        raise ValueError("source text embeds a Qwen chat boundary")
    expected_rendered = rendered_user_prefix(prompt)
    if rendered_prompt != expected_rendered:
        raise ValueError("source rendered_prompt differs from the exact no-system template")
    prefix_ids = normalize_token_ids(tokenizer.encode(rendered_prompt, add_special_tokens=False))
    assistant_ids = normalize_token_ids(tokenizer.encode(response, add_special_tokens=False))
    if not assistant_ids:
        raise ValueError("response tokenized to an empty sequence")
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    im_start_id = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    eos_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    if (eos_id, im_start_id, im_end_id) != (151643, 151644, 151645):
        raise ValueError("Qwen control-token IDs drifted")
    if prefix_ids[0] != im_start_id or prefix_ids.count(im_start_id) != 2:
        raise ValueError("manual Qwen no-system prefix start-token boundary drifted")
    if prefix_ids.count(im_end_id) != 1:
        raise ValueError("manual Qwen no-system prefix end-token boundary drifted")
    if tokenizer.decode(prefix_ids, skip_special_tokens=False) != rendered_prompt:
        raise ValueError("manual Qwen no-system prefix failed round-trip")
    if any(token in assistant_ids for token in (eos_id, im_start_id, im_end_id)):
        raise ValueError("response embeds a Qwen chat-boundary token")
    suffix = [im_end_id, 198]
    input_ids = prefix_ids + assistant_ids + suffix
    if len(input_ids) > max_sequence_length:
        raise ValueError(f"rendered sequence has {len(input_ids)} tokens, cap={max_sequence_length}")
    protected, reasons, literal_positions = protected_assistant_positions(tokenizer, assistant_ids, response)
    result_metadata = dict(metadata)
    result_metadata.update(
        {
            "assistant_token_count": len(assistant_ids),
            "assistant_token_sha256": token_sha256(assistant_ids),
            "loss_weights": [1.0] * len(assistant_ids) + [1.0, 0.0],
            "prefix_token_count": len(prefix_ids),
            "prompt_sha256": sha256_text(prompt),
            "protected_assistant_positions": protected,
            "protected_literal_positions": literal_positions,
            "protected_token_reasons": reasons,
            "rendered_prompt_sha256": sha256_text(rendered_prompt),
            "response_length": len(assistant_ids) + len(suffix),
            "response_token_sha256": token_sha256(assistant_ids),
            "total_token_count": len(input_ids),
            "train_prompt_policy": "source_rendered_prompt_verbatim_one_user_no_system",
            "train_terminal_policy": "assistant_plus_fully_supervised_im_end_newline_weight_zero",
        }
    )
    record = {"input_ids": input_ids, "metadata": result_metadata}
    assert_record(record, max_sequence_length)
    return record


def partition_mixed_problem_ids(problem_ids: Iterable[str], *, seed: int) -> tuple[set[str], set[str]]:
    unique = sorted(set(problem_ids))
    if len(unique) != MIXED_PROBLEM_COUNT:
        raise ValueError(f"partition requires exactly {MIXED_PROBLEM_COUNT} mixed-outcome problems")
    ordered = sorted(
        unique,
        key=lambda problem_id: (
            stable_hex("mixed-problem-split-v1", seed, problem_id),
            problem_id,
        ),
    )
    held_in = set(ordered[:SPLIT_PROBLEM_COUNT])
    held_out = set(ordered[SPLIT_PROBLEM_COUNT:])
    if held_in & held_out or held_in | held_out != set(unique):
        raise AssertionError("mixed-problem partition is not disjoint and exhaustive")
    return held_in, held_out


def select_covered_traces(
    pool: list[dict[str, Any]],
    *,
    problem_ids: set[str],
    target_rows: int,
    seed: int,
    split: str,
    outcome: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select distinct traces while anchoring every explicitly chosen problem."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if outcome not in ("correct", "incorrect"):
        raise ValueError(f"unknown outcome: {outcome}")
    if len(problem_ids) != SPLIT_PROBLEM_COUNT:
        raise ValueError(f"{split}/{outcome} must cover exactly {SPLIT_PROBLEM_COUNT} problems")
    if target_rows < len(problem_ids):
        raise ValueError("cannot cover more problems than selected rows")
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_trace_ids: set[str] = set()
    for row in pool:
        problem_id = str(row["metadata"]["problem_id"])
        trace_id = str(row["metadata"]["base_trace_id"])
        if problem_id not in problem_ids:
            raise ValueError(f"{split}/{outcome} pool contains an unselected problem: {problem_id}")
        if trace_id in seen_trace_ids:
            raise ValueError(f"{split}/{outcome} pool contains a duplicate trace: {trace_id}")
        seen_trace_ids.add(trace_id)
        by_problem[problem_id].append(row)
    if set(by_problem) != problem_ids:
        raise ValueError(f"{split}/{outcome} pool does not cover the exact selected problem set")
    if len(pool) < target_rows:
        raise ValueError(f"{split}/{outcome} has only {len(pool)} traces for target {target_rows}")
    anchors = [
        min(
            by_problem[problem_id],
            key=lambda row: (
                stable_hex(
                    "mixed-control-anchor-v1",
                    seed,
                    split,
                    outcome,
                    row["metadata"]["base_trace_id"],
                ),
                row["metadata"]["base_trace_id"],
            ),
        )
        for problem_id in sorted(problem_ids)
    ]
    anchor_ids = {str(row["metadata"]["base_trace_id"]) for row in anchors}
    remaining = sorted(
        [row for row in pool if str(row["metadata"]["base_trace_id"]) not in anchor_ids],
        key=lambda row: (
            stable_hex(
                "mixed-control-fill-v1",
                seed,
                split,
                outcome,
                row["metadata"]["base_trace_id"],
            ),
            row["metadata"]["base_trace_id"],
        ),
    )
    selected = anchors + remaining[: target_rows - len(anchors)]
    observed_problems = {str(row["metadata"]["problem_id"]) for row in selected}
    if len(selected) != target_rows or observed_problems != problem_ids:
        raise AssertionError(f"{split}/{outcome} covered selection invariant failed")
    selected.sort(key=lambda row: str(row["metadata"]["base_trace_id"]))
    return selected, {
        "algorithm": "one_seeded_anchor_per_fixed_mixed_problem_then_seeded_fill_v1",
        "anchor_rows": len(anchors),
        "fill_rows": target_rows - len(anchors),
        "pool_rows": len(pool),
        "selected_problem_count": len(problem_ids),
        "selected_problem_ids_sha256": ordered_values_sha256(sorted(problem_ids)),
        "selected_rows": len(selected),
        "selected_trace_ids_sha256": ordered_values_sha256(
            str(row["metadata"]["base_trace_id"]) for row in selected
        ),
    }


def expand_balanced(
    records: list[dict[str, Any]], *, variant: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    base_ids = [str(row["metadata"]["base_trace_id"]) for row in records]
    if not base_ids or len(base_ids) != len(set(base_ids)):
        raise ValueError(f"{variant} source trace IDs must be unique and nonempty")
    copies, remainder = divmod(DATASET_ROWS, len(records))
    extra_ids = set(
        sorted(
            base_ids,
            key=lambda trace_id: (
                stable_hex("balanced-6000-extra-v1", seed, variant, trace_id),
                trace_id,
            ),
        )[:remainder]
    )
    expanded = []
    for row in records:
        base_id = str(row["metadata"]["base_trace_id"])
        for copy_index in range(copies + int(base_id in extra_ids)):
            instance = copy.deepcopy(row)
            instance["metadata"].update(
                {
                    "copy_index": copy_index,
                    "extra_physical_copy": copy_index == copies,
                    "trace_id": f"{base_id}:copy_{copy_index:03d}",
                    "variant": variant,
                }
            )
            expanded.append(instance)
    expanded.sort(
        key=lambda row: (
            stable_hex("balanced-6000-order-v1", seed, variant, row["metadata"]["trace_id"]),
            row["metadata"]["trace_id"],
        )
    )
    if len(expanded) != DATASET_ROWS:
        raise AssertionError(f"{variant} did not expand to {DATASET_ROWS} rows")
    exposure = Counter(str(row["metadata"]["base_trace_id"]) for row in expanded)
    if max(exposure.values()) - min(exposure.values()) > 1:
        raise AssertionError(f"{variant} physical source exposure is not balanced")
    return expanded, {
        "base_trace_count": len(records),
        "dataset_rows": len(expanded),
        "effective_runtime_presentations": RUNTIME_PRESENTATIONS,
        "extra_physical_trace_ids_sha256": ordered_values_sha256(sorted(extra_ids)),
        "physical_exposure_histogram": {
            str(count): frequency for count, frequency in sorted(Counter(exposure.values()).items())
        },
        "full_epochs_presentations": DATASET_ROWS * EPOCHS,
        "wrap_presentations": WRAP_PRESENTATIONS,
    }


__all__ = [
    "DATASET_ROWS",
    "EPOCHS",
    "GLOBAL_BATCH_SIZE",
    "MIXED_PROBLEM_COUNT",
    "PROMPT_TEMPLATE",
    "RUNTIME_PRESENTATIONS",
    "SOURCE_TRACE_COUNT",
    "SPLITS",
    "SPLIT_PROBLEM_COUNT",
    "UPDATES",
    "VARIANTS",
    "WRAP_PRESENTATIONS",
    "assert_record",
    "atomic_json",
    "atomic_jsonl",
    "build_record",
    "expand_balanced",
    "length_summary",
    "ordered_values_sha256",
    "partition_mixed_problem_ids",
    "read_jsonl",
    "rendered_user_prefix",
    "select_covered_traces",
    "sha256_file",
    "sha256_text",
    "stable_hex",
    "tokenizer_control_inventory",
]
