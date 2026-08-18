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
EPOCHS = 3
GLOBAL_BATCH_SIZE = 64
UPDATES = math.ceil(DATASET_ROWS * EPOCHS / GLOBAL_BATCH_SIZE)
RUNTIME_PRESENTATIONS = UPDATES * GLOBAL_BATCH_SIZE
WRAP_PRESENTATIONS = RUNTIME_PRESENTATIONS - DATASET_ROWS * EPOCHS
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


def partition_problem_ids(problem_ids: Iterable[str], *, held_in_count: int, seed: int) -> tuple[set[str], set[str]]:
    unique = sorted(set(problem_ids))
    if len(unique) != 480 or held_in_count != 240:
        raise ValueError("partition requires exactly 480 problems and a 240-problem held-in split")
    ordered = sorted(
        unique,
        key=lambda problem_id: (
            stable_hex("qwen3-aime-held-in-partition-v1", seed, problem_id),
            problem_id,
        ),
    )
    held_in = set(ordered[:held_in_count])
    held_out = set(ordered[held_in_count:])
    if held_in & held_out or held_in | held_out != set(unique):
        raise AssertionError("problem partition is not disjoint and exhaustive")
    return held_in, held_out


def select_correct_control(
    correct_pool: list[dict[str, Any]],
    *,
    target_rows: int,
    target_problem_count: int,
    seed: int,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seeded rejection sample of a covered correct control with matched K and N."""
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in correct_pool:
        by_problem[str(row["metadata"]["problem_id"])].append(row)
    if target_rows < target_problem_count:
        raise ValueError("cannot cover more problems than selected correct rows")
    if len(by_problem) < target_problem_count:
        raise ValueError("too few correct-capable problems for the matched-coverage control")
    for attempt in range(10000):
        problem_order = sorted(
            by_problem,
            key=lambda problem_id: (
                stable_hex("correct-control-problems-v1", seed, split, attempt, problem_id),
                problem_id,
            ),
        )
        selected_problem_ids = set(problem_order[:target_problem_count])
        eligible = [row for problem_id in selected_problem_ids for row in by_problem[problem_id]]
        if len(eligible) < target_rows:
            continue
        anchors = [
            min(
                by_problem[problem_id],
                key=lambda row: (
                    stable_hex(
                        "correct-control-anchor-v1",
                        seed,
                        split,
                        attempt,
                        row["metadata"]["base_trace_id"],
                    ),
                    row["metadata"]["base_trace_id"],
                ),
            )
            for problem_id in sorted(selected_problem_ids)
        ]
        anchor_ids = {row["metadata"]["base_trace_id"] for row in anchors}
        remaining = sorted(
            [row for row in eligible if row["metadata"]["base_trace_id"] not in anchor_ids],
            key=lambda row: (
                stable_hex(
                    "correct-control-fill-v1",
                    seed,
                    split,
                    attempt,
                    row["metadata"]["base_trace_id"],
                ),
                row["metadata"]["base_trace_id"],
            ),
        )
        selected = anchors + remaining[: target_rows - len(anchors)]
        observed_problems = {str(row["metadata"]["problem_id"]) for row in selected}
        if len(selected) != target_rows or observed_problems != selected_problem_ids:
            raise AssertionError("correct-control selection invariant failed")
        selected.sort(key=lambda row: str(row["metadata"]["base_trace_id"]))
        return selected, {
            "algorithm": "seeded_problem_subset_rejection_then_one_anchor_each_and_seeded_fill_v1",
            "attempt": attempt,
            "correct_capable_problem_count": len(by_problem),
            "correct_pool_rows": len(correct_pool),
            "selected_problem_count": len(selected_problem_ids),
            "selected_problem_ids_sha256": ordered_values_sha256(sorted(selected_problem_ids)),
            "selected_rows": len(selected),
            "selected_trace_ids_sha256": ordered_values_sha256(
                str(row["metadata"]["base_trace_id"]) for row in selected
            ),
        }
    raise RuntimeError("failed to find a capacity-feasible matched correct-control problem subset")


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
        "three_full_epochs_presentations": DATASET_ROWS * EPOCHS,
        "wrap_presentations": WRAP_PRESENTATIONS,
    }


__all__ = [
    "DATASET_ROWS",
    "EPOCHS",
    "GLOBAL_BATCH_SIZE",
    "PROMPT_TEMPLATE",
    "RUNTIME_PRESENTATIONS",
    "SPLITS",
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
    "partition_problem_ids",
    "read_jsonl",
    "rendered_user_prefix",
    "select_correct_control",
    "sha256_file",
    "sha256_text",
    "stable_hex",
    "tokenizer_control_inventory",
]
