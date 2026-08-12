#!/usr/bin/env python3
"""Prepare deterministic AIME splits and OpenR1 continuation trace sets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    IMAGE_PROBLEM_IDS,
    aime_assistant_text,
    assert_record,
    atomic_json,
    atomic_jsonl,
    build_prompt_record,
    length_summary,
    ordered_values_sha256,
    sha256_file,
    stable_hex,
    tokenizer_control_inventory,
    trace_set_sha256,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    build_training_record,
    normalize_token_ids,
    token_sha256,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    correctness_at,
    trace_structure_ok,
    usable_text,
    value_at,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--aime-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def ensure_outputs_absent(data_root: Path) -> None:
    protected = (
        data_root / "selection_and_tokenization_stats.json",
        data_root / "source_artifacts.json",
        data_root / "tokenizer_control_inventory.json",
        data_root / "selected",
        data_root / "splits",
        data_root / "datasets",
    )
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite preparation artifacts: {existing}")


def load_tokenizer(model: Path, contract: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    for filename, expected in contract["base_model"]["tokenizer_files"].items():
        path = model / filename
        if sha256_file(path) != expected:
            raise ValueError(f"tokenizer source hash drift: {path}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    inventory = tokenizer_control_inventory(tokenizer)
    added = inventory["added_vocabulary"]
    expected_controls = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
    }
    for token, token_id in expected_controls.items():
        if added.get(token) != token_id:
            raise ValueError(f"tokenizer control drift for {token}")
    return tokenizer


def validate_aime_row(row: dict[str, Any]) -> None:
    required = {
        "id",
        "problem_id",
        "year",
        "part",
        "problem_number",
        "question",
        "answer",
        "prompt",
        "reasoning_content",
        "content",
        "correct",
        "model",
    }
    if set(row) != required:
        raise ValueError(
            f"AIME schema drift: missing={sorted(required - set(row))} " f"extra={sorted(set(row) - required)}"
        )
    if not isinstance(row["id"], str) or not isinstance(row["problem_id"], str):
        raise ValueError("AIME id fields must be strings")
    if not isinstance(row["year"], int) or not isinstance(row["problem_number"], int):
        raise ValueError("AIME numeric identity fields must be integers")
    if row["part"] not in (None, "I", "II"):
        raise ValueError("AIME part must be I, II, or null")
    if not isinstance(row["answer"], int) or not 0 <= row["answer"] <= 999:
        raise ValueError("AIME answer must be an integer in [0,999]")
    if row["correct"] is not True or not isinstance(row["model"], str):
        raise ValueError("AIME source must contain only correct model traces")
    for field in ("question", "prompt", "reasoning_content", "content"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"AIME {field} must be nonempty text")
    expected_prompt = row["question"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    if row["prompt"] != expected_prompt:
        raise ValueError("AIME prompt is not exact question plus the audited suffix")
    if last_boxed_integer(row["content"]) != row["answer"]:
        raise ValueError("AIME content last boxed integer does not match answer")


def last_boxed_integer(text: str) -> int | None:
    """Return the last complete balanced ``\\boxed{integer}``, if present."""
    values: list[str] = []
    cursor = 0
    marker = r"\boxed{"
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            values.append(text[content_start : index - 1])
            cursor = index
        else:
            cursor = content_start
    if not values:
        return None
    value = values[-1].strip().replace(",", "")
    return int(value) if value.isdigit() else None


def audit_aime_source(rows: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    for row in rows:
        validate_aime_row(row)
    actual = {
        "all_content_last_boxed_integer_matches_answer": all(
            last_boxed_integer(row["content"]) == row["answer"] for row in rows
        ),
        "all_correct": all(row["correct"] is True for row in rows),
        "all_reasoning_content_nonnull_nonempty": all(
            isinstance(row["reasoning_content"], str) and bool(row["reasoning_content"].strip()) for row in rows
        ),
        "exact_fields": sorted(rows[0]),
        "max_year": max(row["year"] for row in rows),
        "min_year": min(row["year"] for row in rows),
        "problems": len({row["problem_id"] for row in rows}),
        "prompt_is_exact_question_plus_suffix": all(
            row["prompt"]
            == row["question"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
            for row in rows
        ),
        "rows": len(rows),
    }
    if actual != expected:
        raise ValueError(f"AIME source audit drift: expected={expected} actual={actual}")
    return actual


def split_aime_problems(
    rows: list[dict[str, Any]], selection: dict[str, Any]
) -> tuple[list[str], list[str], list[str], dict[str, list[dict[str, Any]]]]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    excluded_year_rows = 0
    excluded_image_rows = 0
    for row in rows:
        validate_aime_row(row)
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate AIME sample ID: {row['id']}")
        seen_ids.add(row["id"])
        if row["year"] == selection["excluded_year"]:
            excluded_year_rows += 1
            continue
        if row["problem_id"] in IMAGE_PROBLEM_IDS:
            excluded_image_rows += 1
            continue
        by_problem[row["problem_id"]].append(row)
    if set(selection["excluded_filename_only_image_problem_ids"]) != IMAGE_PROBLEM_IDS:
        raise ValueError("contract/image exclusion set drift")
    if len(by_problem) != int(selection["expected_eligible_problems"]):
        raise ValueError(
            f"expected {selection['expected_eligible_problems']} AIME problems, "
            f"found {len(by_problem)}; excluded_year_rows={excluded_year_rows} "
            f"excluded_image_rows={excluded_image_rows}"
        )
    for pid, problem_rows in by_problem.items():
        invariants = {
            (
                row["year"],
                row["part"],
                row["problem_number"],
                row["question"],
                row["prompt"],
                row["answer"],
            )
            for row in problem_rows
        }
        if len(invariants) != 1:
            raise ValueError(f"AIME problem metadata drift across trajectories: {pid}")
    problem_ids = sorted(by_problem)
    random.Random(int(selection["seed"])).shuffle(problem_ids)
    held_in_count = int(selection["held_in_problems"])
    held_out_count = int(selection["held_out_problems"])
    held_in = problem_ids[:held_in_count]
    held_out = problem_ids[held_in_count : held_in_count + held_out_count]
    unused = problem_ids[held_in_count + held_out_count :]
    if (len(held_in), len(held_out), len(unused)) != (400, 400, 61):
        raise ValueError("AIME split count drift")
    if set(held_in) & set(held_out) or set(held_in) & set(unused) or set(held_out) & set(unused):
        raise AssertionError("AIME problem splits overlap")
    return held_in, held_out, unused, by_problem


def eval_problem(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "answer": int(row["answer"]),
        "part": row["part"],
        "problem_id": str(row["problem_id"]),
        "problem_number": int(row["problem_number"]),
        "prompt": str(row["prompt"]),
        "question": str(row["question"]),
        "split": split,
        "year": int(row["year"]),
    }


def select_aime_trajectory(
    tokenizer: Any,
    pid: str,
    rows: list[dict[str, Any]],
    seed: int,
    max_sequence_length: int,
    counts: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        if row["correct"] is not True:
            counts["incorrect_source_trace"] += 1
            continue
        try:
            assistant = aime_assistant_text(row["reasoning_content"], row["content"])
            selection_key = stable_hex("aime-trajectory-selection-v1", f"seed={seed}", row["id"])
            metadata = {
                "answer": int(row["answer"]),
                "base_trace_id": str(row["id"]),
                "correctness_source": "sxiong/AIME-trajectory.correct",
                "is_correct": True,
                "problem_id": pid,
                "selection_key_sha256": selection_key,
                "source_dataset": "sxiong/AIME-trajectory",
                "source_model": str(row["model"]),
                "source_sample_id": str(row["id"]),
                "trace_id": str(row["id"]),
            }
            record = build_prompt_record(
                tokenizer,
                user_prompt=str(row["prompt"]),
                assistant_text=assistant,
                metadata=metadata,
                max_sequence_length=max_sequence_length,
            )
        except ValueError as error:
            counts[f"rejected_{str(error).split(':', 1)[0]}"] += 1
            continue
        candidates.append((selection_key, str(row["id"]), row, record))
    if not candidates:
        raise RuntimeError(f"held-in AIME problem has no eligible correct trace: {pid}")
    _, _, source, record = min(candidates, key=lambda item: (item[0], item[1]))
    counts["eligible_selected"] += 1
    selected = {
        "answer": int(source["answer"]),
        "assistant_token_count": int(record["metadata"]["assistant_token_count"]),
        "assistant_token_sha256": str(record["metadata"]["assistant_token_sha256"]),
        "assistant_trace": aime_assistant_text(source["reasoning_content"], source["content"]),
        "content": str(source["content"]),
        "problem_id": pid,
        "prompt": str(source["prompt"]),
        "reasoning_content": str(source["reasoning_content"]),
        "selection_key_sha256": str(record["metadata"]["selection_key_sha256"]),
        "source_model": str(source["model"]),
        "source_sample_id": str(source["id"]),
        "total_token_count": len(record["input_ids"]),
    }
    return selected, record


def audit_aime_rendered_eligibility(
    tokenizer: Any,
    by_problem: dict[str, list[dict[str, Any]]],
    max_sequence_length: int,
    expected: dict[str, Any],
) -> dict[str, Any]:
    eligible_lengths: list[int] = []
    per_problem_minimums: list[int] = []
    over_cap = 0
    other_rejections: Counter[str] = Counter()
    for pid in sorted(by_problem):
        problem_lengths: list[int] = []
        for row in by_problem[pid]:
            try:
                record = build_prompt_record(
                    tokenizer,
                    user_prompt=str(row["prompt"]),
                    assistant_text=aime_assistant_text(row["reasoning_content"], row["content"]),
                    metadata={
                        "is_correct": True,
                        "problem_id": pid,
                        "trace_id": str(row["id"]),
                    },
                    max_sequence_length=max_sequence_length,
                )
            except ValueError as error:
                if str(error).startswith("rendered sequence has "):
                    over_cap += 1
                else:
                    other_rejections[str(error).split(":", 1)[0]] += 1
                continue
            length = len(record["input_ids"])
            eligible_lengths.append(length)
            problem_lengths.append(length)
        if not problem_lengths:
            raise ValueError(f"AIME problem has no fully rendered eligible trace: {pid}")
        per_problem_minimums.append(min(problem_lengths))
    actual = {
        "eligible_problems": len(per_problem_minimums),
        "eligible_rows": len(eligible_lengths),
        "longest_eligible_total_tokens": max(eligible_lengths),
        "max_problem_min_total_tokens": max(per_problem_minimums),
        "other_rejections": dict(other_rejections),
        "over_32768_rows": over_cap,
    }
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"AIME source render audit drift for {key}: " f"expected={value} actual={actual.get(key)}"
            )
    if other_rejections:
        raise ValueError(f"unexpected AIME structural rejections: {other_rejections}")
    return actual


def duplicate_stage1(base_records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], set[str]]:
    if len(base_records) != 400:
        raise ValueError("stage1 needs exactly 400 base trajectories")
    problem_ids = [str(row["metadata"]["problem_id"]) for row in base_records]
    if len(set(problem_ids)) != 400:
        raise ValueError("stage1 base problem IDs are not unique")
    extra = {pid for _, pid in sorted((stable_hex(seed, "aime_extra_copy", pid), pid) for pid in problem_ids)[:200]}
    instances: list[dict[str, Any]] = []
    for source in base_records:
        pid = str(source["metadata"]["problem_id"])
        copies = 8 if pid in extra else 7
        for copy_index in range(copies):
            record = copy.deepcopy(source)
            base_trace_id = str(record["metadata"]["base_trace_id"])
            record["metadata"].update(
                {
                    "copy_index": copy_index,
                    "extra_copy": copy_index == 7,
                    "trace_id": f"{base_trace_id}__copy_{copy_index:02d}",
                    "variant": "aime_correct_3000",
                }
            )
            instances.append(record)
    instances.sort(
        key=lambda row: (
            stable_hex(seed, "aime_stage1_order", row["metadata"]["trace_id"]),
            row["metadata"]["trace_id"],
        )
    )
    if len(instances) != 3000 or len({row["metadata"]["trace_id"] for row in instances}) != 3000:
        raise AssertionError("stage1 duplication did not produce 3,000 unique instances")
    return instances, extra


def openr1_selected(
    source: dict[str, Any],
    row_index: int,
    problem_id: str,
    generation_index: int,
    correct: bool,
    correctness_source: str,
    tokenizer: Any,
    max_sequence_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pid = problem_id
    trace = str(source["generations"][generation_index]).strip()
    assistant_ids = normalize_token_ids(tokenizer.encode(trace, add_special_tokens=False))
    selected = {
        "answer": str(source["answer"]),
        "assistant_token_count": len(assistant_ids),
        "assistant_token_sha256": token_sha256(assistant_ids),
        "assistant_trace": trace,
        "correctness_source": str(correctness_source),
        "generation_index": generation_index,
        "is_correct": correct,
        "problem": str(source["problem"]),
        "problem_id": pid,
        "problem_identity_policy": "sha256_exact_problem_text_utf8",
        "raw_uuid": source.get("uuid"),
        "source": source.get("source"),
        "source_row_index": row_index,
        "trace_id": f"{pid}__row_{row_index:07d}__gen_{generation_index:03d}",
    }
    record = build_training_record(tokenizer, selected, None, max_sequence_length, {})
    record["metadata"].update(
        {
            "base_trace_id": selected["trace_id"],
            "prefix_token_count": len(record["input_ids"]) - int(record["metadata"]["response_length"]),
            "source_dataset": "open-r1/OpenR1-Math-220k",
            "total_token_count": len(record["input_ids"]),
        }
    )
    return selected, record


def eligible_openr1_ref(
    source: dict[str, Any],
    row_index: int,
    problem_id: str,
    correct: bool,
    tokenizer: Any,
    seed: int,
    max_sequence_length: int,
    counts: Counter[str],
) -> dict[str, Any] | None:
    problem = usable_text(source.get("problem"))
    answer = usable_text(source.get("answer"))
    generations = source.get("generations")
    if problem is None or answer is None or not isinstance(generations, (list, tuple)):
        counts["row_missing_required_fields"] += 1
        return None
    pid = problem_id
    candidates: list[tuple[str, int, str]] = []
    for generation_index in range(len(generations)):
        actual, correctness_source = correctness_at(source, generation_index)
        if actual is not correct:
            continue
        if value_at(source.get("is_reasoning_complete"), generation_index) is not True:
            counts["incomplete_source_flag"] += 1
            continue
        trace = usable_text(generations[generation_index])
        if trace is None:
            counts["empty_trace"] += 1
            continue
        structure_ok, reason = trace_structure_ok(trace)
        if not structure_ok:
            counts[f"rejected_{reason}"] += 1
            continue
        key = stable_hex(
            seed,
            f"openr1_{'correct' if correct else 'wrong'}_trajectory",
            pid,
            generation_index,
        )
        candidates.append((key, generation_index, str(correctness_source)))
    for key, generation_index, correctness_source in sorted(candidates):
        try:
            _, record = openr1_selected(
                source,
                row_index,
                pid,
                generation_index,
                correct,
                correctness_source,
                tokenizer,
                max_sequence_length,
            )
        except ValueError:
            counts["rendered_sequence_rejected"] += 1
            continue
        counts["eligible_problem"] += 1
        return {
            "assistant_tokens": int(record["metadata"]["assistant_token_count"]),
            "correctness_source": correctness_source,
            "generation_index": generation_index,
            "problem_id": pid,
            "response_tokens": int(record["metadata"]["response_length"]),
            "source_row_index": row_index,
            "total_tokens": len(record["input_ids"]),
            "trajectory_key_sha256": key,
        }
    counts["problem_without_eligible_trace"] += 1
    return None


def semantic_openr1_problem_id(source: dict[str, Any]) -> tuple[str, str]:
    """Return SHA-256 identity of exact problem text plus that validated text."""
    problem = source.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("OpenR1 problem must be nonempty text")
    return hashlib.sha256(problem.encode("utf-8")).hexdigest(), problem


def normalized_question_sha256(text: str) -> str:
    canonical = " ".join(text.split())
    if not canonical:
        raise ValueError("question text is empty after whitespace normalization")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def select_openr1_refs(
    dataset: Any,
    tokenizer: Any,
    seed: int,
    target: int,
    max_sequence_length: int,
    expected_source_id_audit: dict[str, Any],
    excluded_aime_question_splits: dict[str, str],
    expected_aime_overlap: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidate_row_by_problem: dict[str, int] = {}
    correct_counts: Counter[str] = Counter()
    wrong_counts: Counter[str] = Counter()
    raw_id_counts: Counter[str] = Counter()
    problem_text_by_id: dict[str, str] = {}
    problem_text_counts: Counter[str] = Counter()
    aime_overlap_counts: Counter[str] = Counter()
    for row_index, source in enumerate(dataset):
        raw_id_counts[str(source.get("uuid")).strip()] += 1
        pid, exact_problem = semantic_openr1_problem_id(source)
        if pid in problem_text_by_id and problem_text_by_id[pid] != exact_problem:
            raise RuntimeError("SHA-256 collision between distinct OpenR1 problem texts")
        problem_text_by_id[pid] = exact_problem
        problem_text_counts[exact_problem] += 1
        question_hash = normalized_question_sha256(exact_problem)
        if question_hash in excluded_aime_question_splits:
            aime_overlap_counts[excluded_aime_question_splits[question_hash]] += 1
            continue
        if pid in candidate_row_by_problem:
            raise ValueError("OpenR1 semantic problem identity is not source-unique")
        candidate_row_by_problem[pid] = row_index
        if (row_index + 1) % 5000 == 0:
            print(
                "OPENR1_ELIGIBILITY_PROGRESS "
                f"rows={row_index + 1} non_aime_candidates={len(candidate_row_by_problem)}",
                flush=True,
            )
    source_id_audit = {
        "duplicate_raw_id_values": sum(count > 1 for count in raw_id_counts.values()),
        "duplicate_valid_raw_id_values": sum(
            count > 1
            for value, count in raw_id_counts.items()
            if value.strip().lower() not in {"", "none", "null", "nan", "[]"}
        ),
        "invalid_raw_id_rows": sum(
            count
            for value, count in raw_id_counts.items()
            if value.strip().lower() in {"", "none", "null", "nan", "[]"}
        ),
        "raw_unique_id_values": len(raw_id_counts),
        "rows": sum(raw_id_counts.values()),
        "semantic_problem_id_policy": "sha256_exact_problem_text_utf8",
        "unique_problem_texts": len(problem_text_counts),
    }
    for key in (
        "duplicate_raw_id_values",
        "duplicate_valid_raw_id_values",
        "invalid_raw_id_rows",
        "raw_unique_id_values",
        "rows",
        "unique_problem_texts",
    ):
        expected = expected_source_id_audit[key]
        if source_id_audit[key] != expected:
            raise ValueError(
                f"OpenR1 source ID audit drift for {key}: " f"expected={expected} actual={source_id_audit[key]}"
            )
    actual_aime_overlap = {
        "held_in_overlaps": aime_overlap_counts["held_in"],
        "held_out_problem_overlaps": aime_overlap_counts["held_out_problem"],
        "source_overlaps": sum(aime_overlap_counts.values()),
        "unused_overlaps": aime_overlap_counts["unused"],
    }
    for key, actual in actual_aime_overlap.items():
        if actual != expected_aime_overlap[key]:
            raise ValueError(
                f"OpenR1/AIME overlap audit drift for {key}: " f"expected={expected_aime_overlap[key]} actual={actual}"
            )

    # Eligibility is expensive because it renders and tokenizes traces.  Rank the
    # source-unique semantic problem IDs first, then scan that exact order until
    # the requested number of eligible problems is reached.  This is equivalent
    # to selecting the minimum problem keys among the fully eligible population.
    wrong: list[dict[str, Any]] = []
    wrong_examined = 0
    for pid, row_index in sorted(
        candidate_row_by_problem.items(),
        key=lambda item: (
            stable_hex(seed, "openr1_wrong_problem", item[0]),
            item[0],
        ),
    ):
        wrong_examined += 1
        ref = eligible_openr1_ref(
            dataset[row_index],
            row_index,
            pid,
            False,
            tokenizer,
            seed,
            max_sequence_length,
            wrong_counts,
        )
        if ref is not None:
            wrong.append(ref)
        if len(wrong) == target:
            break
    wrong_problem_ids = {str(ref["problem_id"]) for ref in wrong}
    correct: list[dict[str, Any]] = []
    correct_examined = 0
    for pid, row_index in sorted(
        (item for item in candidate_row_by_problem.items() if item[0] not in wrong_problem_ids),
        key=lambda item: (
            stable_hex(seed, "openr1_correct_problem", item[0]),
            item[0],
        ),
    ):
        correct_examined += 1
        ref = eligible_openr1_ref(
            dataset[row_index],
            row_index,
            pid,
            True,
            tokenizer,
            seed,
            max_sequence_length,
            correct_counts,
        )
        if ref is not None:
            correct.append(ref)
        if len(correct) == target:
            break
    if len(wrong) != target or len(correct) != target:
        raise RuntimeError(f"insufficient OpenR1 eligibility correct={len(correct)} wrong={len(wrong)}")
    if wrong_problem_ids & {str(ref["problem_id"]) for ref in correct}:
        raise AssertionError("OpenR1 correct/wrong problem IDs overlap")
    return (
        correct,
        wrong,
        {
            "aime_question_exclusion_audit": actual_aime_overlap,
            "correct_candidate_problems_examined": correct_examined,
            "correct_eligible_problems_selected": len(correct),
            "correct_skip_counts": dict(correct_counts),
            "source_id_audit": source_id_audit,
            "wrong_candidate_problems_examined": wrong_examined,
            "wrong_eligible_problems_selected": len(wrong),
            "wrong_skip_counts": dict(wrong_counts),
        },
    )


def materialize_openr1(
    dataset: Any,
    refs: list[dict[str, Any]],
    correct: bool,
    tokenizer: Any,
    max_sequence_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, ref in enumerate(refs):
        source = dataset[int(ref["source_row_index"])]
        selected, record = openr1_selected(
            source,
            int(ref["source_row_index"]),
            str(ref["problem_id"]),
            int(ref["generation_index"]),
            correct,
            str(ref["correctness_source"]),
            tokenizer,
            max_sequence_length,
        )
        selected["selection_index"] = index
        selected["trajectory_key_sha256"] = ref["trajectory_key_sha256"]
        record["metadata"]["selection_index"] = index
        record["metadata"]["variant"] = "continue_correct_only_1500" if correct else "continue_wrong_unmasked"
        assert_record(record, max_sequence_length)
        selected_rows.append(selected)
        records.append(record)
    return selected_rows, records


def write_split(
    path: Path,
    problem_ids: list[str],
    by_problem: dict[str, list[dict[str, Any]]],
    split: str,
) -> list[dict[str, Any]]:
    rows = [eval_problem(by_problem[pid][0], split) for pid in problem_ids]
    atomic_jsonl(path, rows)
    return rows


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["family"] != "qwen2_5_7b_aime_generalization_incorrect_sft":
        raise ValueError("wrong experiment contract")
    ensure_outputs_absent(args.data_root)
    selection = contract["selection"]
    aime_spec = selection["aime"]
    if sha256_file(args.aime_source) != aime_spec["dataset"]["source_jsonl_sha256"]:
        raise ValueError("AIME source JSONL hash drift")
    tokenizer = load_tokenizer(args.model, contract)
    max_sequence_length = int(contract["training"]["max_sequence_length"])
    seed = int(aime_spec["seed"])

    aime_rows: list[dict[str, Any]] = []
    with args.aime_source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                aime_rows.append(json.loads(line))
    source_audit = audit_aime_source(aime_rows, aime_spec["source_audit"])
    held_in, held_out, unused, by_problem = split_aime_problems(aime_rows, aime_spec)
    source_render_audit = audit_aime_rendered_eligibility(
        tokenizer,
        by_problem,
        max_sequence_length,
        aime_spec["source_render_audit"],
    )
    aime_question_splits: dict[str, str] = {}
    for split, problem_ids in (
        ("held_in", held_in),
        ("held_out_problem", held_out),
        ("unused", unused),
    ):
        for pid in problem_ids:
            question_hash = normalized_question_sha256(str(by_problem[pid][0]["question"]))
            if question_hash in aime_question_splits:
                raise ValueError("eligible AIME questions are not text-unique")
            aime_question_splits[question_hash] = split
    split_root = args.data_root / "splits"
    held_in_eval = write_split(split_root / "held_in_400.jsonl", held_in, by_problem, "held_in")
    held_out_eval = write_split(
        split_root / "held_out_problem_400.jsonl",
        held_out,
        by_problem,
        "held_out_problem",
    )
    write_split(split_root / "unused_61.jsonl", unused, by_problem, "unused")

    aime_counts: Counter[str] = Counter()
    aime_selected: list[dict[str, Any]] = []
    aime_base_records: list[dict[str, Any]] = []
    for pid in held_in:
        selected, record = select_aime_trajectory(
            tokenizer,
            pid,
            by_problem[pid],
            seed,
            max_sequence_length,
            aime_counts,
        )
        assert_record(record, max_sequence_length)
        aime_selected.append(selected)
        aime_base_records.append(record)
    stage1_records, extra_problem_ids = duplicate_stage1(aime_base_records, seed)
    for record in stage1_records:
        assert_record(record, max_sequence_length)

    from datasets import load_dataset

    openr1_spec = selection["openr1"]
    dataset_spec = openr1_spec["dataset"]
    openr1_dataset = load_dataset(
        dataset_spec["hf_repo"],
        dataset_spec["config"],
        split=dataset_spec["split"],
        revision=dataset_spec["revision"],
        cache_dir=str(args.cache_dir),
    )
    correct_refs, wrong_refs, openr1_eligibility = select_openr1_refs(
        openr1_dataset,
        tokenizer,
        int(openr1_spec["seed"]),
        int(openr1_spec["correct_rows"]),
        max_sequence_length,
        openr1_spec["source_id_audit"],
        aime_question_splits,
        openr1_spec["aime_question_exclusion"],
    )
    correct_selected, correct_records = materialize_openr1(
        openr1_dataset,
        correct_refs,
        True,
        tokenizer,
        max_sequence_length,
    )
    wrong_selected, wrong_records = materialize_openr1(
        openr1_dataset,
        wrong_refs,
        False,
        tokenizer,
        max_sequence_length,
    )
    correct_problem_ids = {row["problem_id"] for row in correct_selected}
    wrong_problem_ids = {row["problem_id"] for row in wrong_selected}
    if len(correct_problem_ids) != 1500 or len(wrong_problem_ids) != 1500:
        raise ValueError("OpenR1 selection is not one unique problem per row")
    if correct_problem_ids & wrong_problem_ids:
        raise ValueError("OpenR1 correct and wrong problem IDs overlap")
    selected_question_hashes = {
        normalized_question_sha256(str(row["problem"])) for row in correct_selected + wrong_selected
    }
    if selected_question_hashes & set(aime_question_splits):
        raise ValueError("selected OpenR1 continuation directly overlaps AIME")

    selected_root = args.data_root / "selected"
    datasets_root = args.data_root / "datasets"
    selected_paths = {
        "aime": selected_root / "aime_held_in_correct_400.jsonl",
        "correct": selected_root / "openr1_correct_1500.jsonl",
        "wrong": selected_root / "openr1_wrong_1500.jsonl",
    }
    dataset_paths = {
        "stage1": datasets_root / "aime_correct_3000.jsonl",
        "correct": datasets_root / "continue_correct_only_1500.jsonl",
        "wrong": datasets_root / "continue_wrong_unmasked_1500.jsonl",
    }
    atomic_jsonl(selected_paths["aime"], aime_selected)
    atomic_jsonl(selected_paths["correct"], correct_selected)
    atomic_jsonl(selected_paths["wrong"], wrong_selected)
    atomic_jsonl(dataset_paths["stage1"], stage1_records)
    atomic_jsonl(dataset_paths["correct"], correct_records)
    atomic_jsonl(dataset_paths["wrong"], wrong_records)
    inventory_path = args.data_root / "tokenizer_control_inventory.json"
    atomic_json(inventory_path, tokenizer_control_inventory(tokenizer))

    artifacts = {
        "aime_source": {
            "path": str(args.aime_source.resolve()),
            "sha256": sha256_file(args.aime_source),
        },
        "datasets": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in dataset_paths.items()
        },
        "selected": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in selected_paths.items()
        },
        "splits": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in {
                "held_in": split_root / "held_in_400.jsonl",
                "held_out_problem": split_root / "held_out_problem_400.jsonl",
                "unused": split_root / "unused_61.jsonl",
            }.items()
        },
        "tokenizer_inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
        },
    }
    source_artifacts_path = args.data_root / "source_artifacts.json"
    atomic_json(source_artifacts_path, artifacts)
    stats = {
        "artifact_schema_version": 1,
        "aime": {
            "base_selected_length_stats": length_summary(aime_base_records),
            "extra_copy_problem_ids": sorted(extra_problem_ids),
            "extra_copy_problem_ids_sha256": ordered_values_sha256(sorted(extra_problem_ids)),
            "held_in_eval_rows": len(held_in_eval),
            "held_in_problem_ids_sha256": ordered_values_sha256(held_in),
            "held_out_eval_rows": len(held_out_eval),
            "held_out_problem_ids_sha256": ordered_values_sha256(held_out),
            "selected_trace_ids_sha256": trace_set_sha256([row["source_sample_id"] for row in aime_selected]),
            "selection_counts": dict(aime_counts),
            "source_audit": source_audit,
            "source_render_audit": source_render_audit,
            "source_problems": len({row["problem_id"] for row in aime_rows}),
            "source_rows": len(aime_rows),
            "stage1_length_stats": length_summary(stage1_records),
            "stage1_rows": len(stage1_records),
            "stage1_trace_ids_sha256": trace_set_sha256([row["metadata"]["trace_id"] for row in stage1_records]),
            "unused_problem_ids_sha256": ordered_values_sha256(unused),
        },
        "invariants": {
            "aime_prompt_field_used_verbatim": True,
            "aime_stage1_200_problems_have_eight_copies": True,
            "aime_stage1_200_problems_have_seven_copies": True,
            "all_rendered_sequences_at_most_32768": True,
            "correct_and_wrong_openr1_problem_ids_disjoint": True,
            "no_synthetic_endoftext": True,
            "openr1_one_trace_per_unique_problem": True,
            "terminal_im_end_weight_one_newline_weight_zero": True,
        },
        "openr1": {
            "correct_length_stats": length_summary(correct_records),
            "correct_problem_ids_sha256": ordered_values_sha256([row["problem_id"] for row in correct_selected]),
            "correct_trace_ids_sha256": trace_set_sha256([row["trace_id"] for row in correct_selected]),
            "eligibility": openr1_eligibility,
            "wrong_length_stats": length_summary(wrong_records),
            "wrong_problem_ids_sha256": ordered_values_sha256([row["problem_id"] for row in wrong_selected]),
            "wrong_trace_ids_sha256": trace_set_sha256([row["trace_id"] for row in wrong_selected]),
        },
        "source_artifacts_sha256": sha256_file(source_artifacts_path),
    }
    stats_path = args.data_root / "selection_and_tokenization_stats.json"
    atomic_json(stats_path, stats)
    print(
        "AIME_GENERALIZATION_PREP_COMPLETE "
        f"held_in={len(held_in)} held_out={len(held_out)} stage1={len(stage1_records)} "
        f"correct={len(correct_records)} wrong={len(wrong_records)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
