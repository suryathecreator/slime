#!/usr/bin/env python3
"""Redacted Llama span gate and strict MATH-500 symbolic rescoring.

The GPU gate consumes only ``gate_inputs.jsonl`` shards.  Gold answers live in
a separate scoring directory and are first opened by the CPU scoring command.
Llama adjudicates answer state and selects source-span IDs; it never generates
answer text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from examples.qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff import (
    axolotl_math500_eval as native,
)


HANDOFF = Path(__file__).resolve().parent
POLICY_PATH = HANDOFF / "llama_gate_policy.json"
GATE_ARTIFACT_VERSION = 2
GATE_IMPLEMENTATION_VERSION = "llama-span-gate-v2.0.1"
SPAN_CATALOG_VERSION = "latex-lexer-v2.0.1"
SCORER_VERSION = "strict-router-v2.0.0"
METRIC_NAME = "math500_llama33_definitive_strict_v2"
EXPECTED_VARIANTS = 12
EXPECTED_SHARDS = 4
ROWS_PER_SHARD = 125
EXPECTED_ROWS = EXPECTED_VARIANTS * EXPECTED_SHARDS * ROWS_PER_SHARD
ALLOWED_GATE_INPUT_FIELDS = {
    "problem",
    "problem_id",
    "response",
    "response_sha256",
    "shard_index",
    "variant",
}
ANSWER_FORMS = {
    "scalar",
    "expression",
    "equation",
    "tuple_or_list",
    "set_or_interval",
    "multiple_choice",
    "other",
    "none",
}
COMMITTED_REASONS = {
    "single_committed_answer",
    "single_committed_unboxed_answer",
    "equivalent_forms_committed",
}
NONCOMMITTED_REASONS = {
    "no_answer",
    "only_intermediate_result",
    "incomplete_answer",
    "unresolved_hedge",
    "multiple_unresolved_answers",
    "answer_retracted",
    "answer_later_questioned",
    "ambiguous_final_commitment",
}
TERMINAL_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "definitive",
        "final_answer_raw",
        "boxed_final_answer",
        "format_compliant",
        "answer_form",
        "supporting_quote",
        "later_conflict_quote",
        "reason_code",
    ],
    "properties": {
        "definitive": {"type": "boolean"},
        "final_answer_raw": {"type": ["string", "null"]},
        "boxed_final_answer": {"type": "boolean"},
        "format_compliant": {"type": "boolean"},
        "answer_form": {"type": "string", "enum": sorted(ANSWER_FORMS)},
        "supporting_quote": {"type": ["string", "null"]},
        "later_conflict_quote": {"type": ["string", "null"]},
        "reason_code": {
            "type": "string",
            "enum": sorted(COMMITTED_REASONS | NONCOMMITTED_REASONS),
        },
    },
}
ADJUDICATION_FIELDS = {
    "definitive",
    "boxed_final_answer",
    "format_compliant",
    "answer_form",
    "evidence_region_id",
    "reason_code",
}
EXTRACTION_FIELDS = {
    "extraction_mode",
    "candidate_id",
    "start_boundary_id",
    "end_boundary_id",
}
TEXT_GOLD_IDS = {
    "test/algebra/1349.json",
    "test/prealgebra/105.json",
    "test/intermediate_algebra/128.json",
    "test/precalculus/819.json",
    "test/precalculus/452.json",
    "test/intermediate_algebra/754.json",
    "test/intermediate_algebra/860.json",
    "test/prealgebra/1991.json",
}


def policy() -> dict[str, Any]:
    return native.controller.load_json(POLICY_PATH)


def gate_policy_manifest() -> dict[str, Any]:
    """Identity every input that can affect adjudication or span selection."""
    gate_policy = policy()
    prompt_names = (
        str(gate_policy["gate"]["adjudication_prompt"]),
        str(gate_policy["gate"]["extraction_prompt"]),
    )
    gate_runtime_names = {"llguidance", "torch", "transformers", "vllm", "xgrammar"}
    return {
        "adjudication_fields": sorted(ADJUDICATION_FIELDS),
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "extraction_fields": sorted(EXTRACTION_FIELDS),
        "gate_implementation_version": GATE_IMPLEMENTATION_VERSION,
        "policy": {
            "engine": gate_policy["engine"],
            "gate": gate_policy["gate"],
            "model": gate_policy["model"],
            "runtime": {
                name: version
                for name, version in gate_policy["runtime"].items()
                if name in gate_runtime_names
            },
        },
        "prompt_sha256": {
            name: native.controller.sha256_file(HANDOFF / name) for name in prompt_names
        },
        "span_catalog_version": SPAN_CATALOG_VERSION,
        "terminal_decision_schema": TERMINAL_DECISION_SCHEMA,
    }


def gate_policy_sha256() -> str:
    return sha256_text(native.controller.canonical_json(gate_policy_manifest()))


def inventory(repo_root: Path) -> dict[str, Any]:
    # load_inventory's default was bound when the shared Qwen3 controller was
    # defined, so pass the Qwen2 handoff manifest explicitly.
    return native.controller.load_inventory(repo_root, native.controller.DEFAULT_MANIFEST)


def roots(repo_root: Path) -> dict[str, Path]:
    value = inventory(repo_root)
    evaluation = native.controller.eval_root(repo_root, value)
    no_gold = evaluation / "_llama_gate_v2_no_gold"
    scoring = evaluation / "_llama_strict_scoring_v2"
    runtime = repo_root / "checkpoints/runtime/qwen25-math500-llama-gate-v1/venv"
    model = (
        repo_root
        / "checkpoints/models/nvidia--Llama-3.3-70B-Instruct-FP8--68579f675008"
    )
    return {
        "native_control": native.controller.control_root(repo_root, value),
        "gate": no_gold,
        "gate_inputs": no_gold / "gate_inputs",
        "gate_outputs": no_gold / "gate_outputs",
        "gate_metadata": no_gold / "metadata.json",
        "gate_overlays": no_gold / "gate_overlays",
        "canary_inputs": no_gold / "canary_inputs.jsonl",
        "canary_outputs": no_gold / "canary_outputs.jsonl",
        "canary_complete": no_gold / "canary.complete.json",
        "gold": scoring / "gold_structure_manifest.jsonl",
        "scoring": scoring,
        "scores": scoring / "scores.jsonl",
        "suite_result": scoring / "RESULTS.json",
        "runtime": runtime,
        "python": runtime / "bin/python",
        "model": model,
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_row_key(
    variant: str,
    problem_id: str,
    response_sha256: str,
    policy_sha256: str,
) -> str:
    return sha256_text(
        "\0".join((variant, problem_id, response_sha256, policy_sha256))
    )


def variants(value: dict[str, Any]) -> list[str]:
    result = [str(item["variant"]) for item in value["checkpoints"]]
    if len(result) != EXPECTED_VARIANTS or len(set(result)) != EXPECTED_VARIANTS:
        raise RuntimeError("expected exactly 12 unique evaluation variants")
    return result


def input_path(repo_root: Path, variant: str, shard_index: int) -> Path:
    return roots(repo_root)["gate_inputs"] / variant / f"shard_{shard_index:02d}.jsonl"


def decision_path(repo_root: Path, variant: str, shard_index: int) -> Path:
    return roots(repo_root)["gate_outputs"] / variant / f"shard_{shard_index:02d}.jsonl"


def decision_complete_path(repo_root: Path, variant: str, shard_index: int) -> Path:
    return roots(repo_root)["gate_outputs"] / variant / f"shard_{shard_index:02d}.complete.json"


def result_path(repo_root: Path, item: dict[str, Any]) -> Path:
    native_result = native.controller.absolute_item_paths(repo_root, item)["result"]
    return native_result / "rescored" / METRIC_NAME


def validate_model(repo_root: Path) -> dict[str, Any]:
    model_root = roots(repo_root)["model"]
    ready = native.controller.load_json(model_root / "MODEL_READY.json")
    expected_model = policy()["model"]
    if {key: ready.get(key) for key in ("hf_repo", "revision")} != expected_model:
        raise RuntimeError("local Llama model readiness pin mismatch")
    index = native.controller.load_json(model_root / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 15 or any(not (model_root / shard).is_file() for shard in shards):
        raise RuntimeError("pinned Llama model does not have 15 complete weight shards")
    quantization = native.controller.load_json(model_root / "hf_quant_config.json")
    if quantization.get("producer", {}).get("name") != "modelopt":
        raise RuntimeError("pinned Llama model is not ModelOpt-produced")
    quantization_values = quantization.get("quantization", {})
    if quantization_values.get("quant_algo") != "FP8":
        raise RuntimeError("pinned Llama weights are not FP8")
    if quantization_values.get("kv_cache_quant_algo") != "FP8":
        raise RuntimeError("pinned Llama model does not declare FP8 KV cache")
    return ready


def strip_thousands(value: str) -> str:
    previous = None
    result = value
    pattern = re.compile(r"(?<=\d),(?:\\!\s*|\s*)(?=\d{3}(?:\D|$))")
    while previous != result:
        previous, result = result, pattern.sub("", result)
    return result


def split_top_level(value: str, separator: str = ",") -> list[str]:
    """Split LaTeX at separators that are outside (), [], and {} groups."""
    parts: list[str] = []
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    start = 0
    index = 0
    while index < len(value):
        character = value[index]
        if not stack and value.startswith(separator, index):
            parts.append(value[start:index].strip())
            index += len(separator)
            start = index
            continue
        if character == "\\":
            index += 2
            continue
        if character in "([{":
            stack.append(character)
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
        index += 1
    parts.append(value[start:].strip())
    return parts


def unwrap_outer(value: str, opening: str, closing: str) -> str | None:
    cleaned = value.strip()
    cleaned = re.sub(r"\\(?:left|right)\s*", "", cleaned)
    if not (cleaned.startswith(opening) and cleaned.endswith(closing)):
        return None
    stack = 0
    for index, character in enumerate(cleaned):
        if character == opening:
            stack += 1
        elif character == closing:
            stack -= 1
            if stack == 0 and index != len(cleaned) - 1:
                return None
    return cleaned[1:-1].strip() if stack == 0 else None


def expand_plus_minus(parts: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for part in parts:
        if "\\pm" in part:
            expanded.extend((part.replace("\\pm", "+", 1), part.replace("\\pm", "-", 1)))
        elif "±" in part:
            expanded.extend((part.replace("±", "+", 1), part.replace("±", "-", 1)))
        else:
            expanded.append(part)
    return expanded


def tuple_parts(value: str) -> list[str] | None:
    inner = unwrap_outer(strip_thousands(value), "(", ")")
    if inner is None:
        return None
    parts = split_top_level(inner)
    return parts if len(parts) > 1 and all(parts) else None


def set_parts(value: str, allow_bare: bool = True) -> list[str] | None:
    cleaned = strip_thousands(value).strip()
    inner = None
    if cleaned.startswith(r"\{") and cleaned.endswith(r"\}"):
        inner = cleaned[2:-2].strip()
    elif cleaned.startswith("{") and cleaned.endswith("}"):
        inner = unwrap_outer(cleaned, "{", "}")
    elif allow_bare:
        inner = cleaned
    if inner is None:
        return None
    parts = expand_plus_minus(split_top_level(inner))
    return parts if len(parts) > 1 and all(parts) else None


def interval_parts(value: str) -> list[dict[str, str]] | None:
    cleaned = value.strip()
    membership = re.match(r"^[A-Za-z]+\s*\\in\s*(.+)$", cleaned)
    if membership:
        cleaned = membership.group(1).strip()
    unions = split_top_level(cleaned, r"\cup")
    result: list[dict[str, str]] = []
    for component in unions:
        simple = re.sub(r"\\(?:left|right)\s*", "", component.strip())
        if len(simple) < 3 or simple[0] not in "([" or simple[-1] not in ")]":
            return None
        endpoints = split_top_level(simple[1:-1])
        if len(endpoints) != 2 or not all(endpoints):
            return None
        result.append(
            {
                "left_boundary": simple[0],
                "left_endpoint": endpoints[0],
                "right_endpoint": endpoints[1],
                "right_boundary": simple[-1],
            }
        )
    return result or None


def equation_parts(value: str) -> list[str] | None:
    parts = split_top_level(value, "=")
    return parts if len(parts) == 2 and all(parts) else None


def classify_gold(problem_id: str, problem: str, answer: str) -> dict[str, Any]:
    if problem_id in TEXT_GOLD_IDS:
        kind = "multiple_choice" if re.search(r"\([A-E]\)", answer) else "text"
        return {"gold_type": kind, "parts": [answer]}
    lower_problem = problem.lower()
    intervals = interval_parts(answer)
    interval_notation = re.sub(r"\\(?:left|right)\s*", "", answer.strip())
    interval_evidence = (
        r"\infty" in answer
        or r"\cup" in answer
        or (
            intervals is not None
            and (interval_notation[:1] == "[") != (interval_notation[-1:] == "]")
        )
        or (intervals is not None and interval_notation[:1] == "[")
        or "interval notation" in lower_problem
        or answer.lstrip().startswith(("x \\in", "x\\in"))
        or ("range" in lower_problem and intervals is not None)
        or ("domain" in lower_problem and intervals is not None)
    )
    if intervals is not None and interval_evidence:
        return {"gold_type": "interval", "parts": intervals}
    equation = equation_parts(answer)
    if equation is not None:
        return {"gold_type": "equation", "parts": equation}
    tuple_value = tuple_parts(answer)
    tuple_evidence = any(
        phrase in lower_problem
        for phrase in (
            "ordered pair",
            "ordered triple",
            "ordered quadruple",
            "coordinates",
            "resulting point",
            "find the point",
            "point the curve",
            "midpoint",
            "polar coordinates",
        )
    )
    if tuple_value is not None and tuple_evidence:
        return {"gold_type": "ordered_tuple", "parts": tuple_value}
    finite = set_parts(answer)
    if finite is not None:
        return {"gold_type": "finite_set", "parts": finite}
    return {"gold_type": "math", "parts": [answer]}


def classify_gold_routes(problem_id: str, problem: str, answer: str) -> list[dict[str, Any]]:
    """Return the heuristic route first, followed by conservative alternatives."""
    primary = classify_gold(problem_id, problem, answer)
    routes = [primary]

    def add(kind: str, parts: Any) -> None:
        route = {"gold_type": kind, "parts": parts}
        if route not in routes:
            routes.append(route)

    if problem_id in TEXT_GOLD_IDS:
        return routes
    intervals = interval_parts(answer)
    if intervals is not None:
        add("interval", intervals)
    equation = equation_parts(answer)
    if equation is not None:
        add("equation", equation)
    tuple_value = tuple_parts(answer)
    if tuple_value is not None:
        add("ordered_tuple", tuple_value)
    finite = set_parts(answer, allow_bare=tuple_value is None and intervals is None)
    if finite is not None:
        add("finite_set", finite)
    if all(value is None for value in (intervals, equation, tuple_value, finite)):
        add("math", [answer])
    return routes


def select_canary_rows(repo_root: Path, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Include every known v1 failure, then deterministic edge-case representatives."""
    by_identity = {
        (str(row["variant"]), str(row["problem_id"])): row for row in rows
    }
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()

    def add(row: dict[str, Any]) -> None:
        key = (str(row["variant"]), str(row["problem_id"]))
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)

    evaluation = native.controller.eval_root(repo_root, inventory(repo_root))
    legacy_failures = (
        evaluation / "_llama_gate_no_gold/gate_outputs/_validation_failures"
    )
    if legacy_failures.is_dir():
        for failure_path in sorted(legacy_failures.glob("**/*.jsonl")):
            variant = failure_path.parent.name
            for failure in native.controller.read_jsonl(failure_path):
                source = by_identity.get((variant, str(failure.get("problem_id"))))
                if source is not None:
                    add(source)

    predicates = (
        lambda text: text.count(r"\boxed") > 1,
        lambda text: r"\boxed" not in text and bool(re.search(r"final answer|answer is", text, re.I)),
        lambda text: bool(re.search(r"\boxed\s*\{\s*\(?[^\n]*,[^\n]*\)?\s*\}", text)),
        lambda text: bool(re.search(r"\boxed\s*\{[^\n]*(?:\\infty|\\cup)[^\n]*\}", text)),
        lambda text: bool(re.search(r"\b(?:however|but|instead|correction)\b", text, re.I)),
        lambda text: len(text) > 12000,
    )
    for predicate in predicates:
        source = next((row for row in rows if predicate(str(row["response"]))), None)
        if source is not None:
            add(source)
    for row in sorted(rows, key=lambda value: len(str(value["response"])), reverse=True):
        add(row)
        if len(selected) >= 20:
            break
    return selected


def prepare_inputs(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    value = inventory(repo_root)
    paths = roots(repo_root)
    audit = paths["native_control"] / "FINAL_AUDIT.json"
    if not audit.is_file() or native.controller.load_json(audit).get("status") != "complete":
        raise RuntimeError("redaction requires the complete native FINAL_AUDIT.json")
    benchmark, shard_files, _ = native.controller.benchmark_paths(repo_root, value)
    benchmark_rows = native.controller.read_jsonl(benchmark)
    if len(benchmark_rows) != 500:
        raise RuntimeError("canonical MATH-500 must contain exactly 500 rows")
    benchmark_by_id = {str(row["problem_id"]): row for row in benchmark_rows}
    policy_sha = gate_policy_sha256()
    input_records: list[dict[str, Any]] = []
    gold_records: list[dict[str, Any]] = []
    for row in benchmark_rows:
        problem_id = str(row["problem_id"])
        problem = str(row["problem"])
        correct_answer = str(row["correct_answer"])
        routes = classify_gold_routes(problem_id, problem, correct_answer)
        structure = routes[0]
        gold_records.append(
            {
                "correct_answer": correct_answer,
                "problem": problem,
                "problem_id": problem_id,
                "route_candidates": routes,
                **structure,
            }
        )
    for item in value["checkpoints"]:
        variant = str(item["variant"])
        native_paths = native.controller.absolute_item_paths(repo_root, item)
        raw_path = native_paths["output"] / "merged/raw_generations.jsonl"
        raw_rows = native.controller.read_jsonl(raw_path)
        raw_by_id = {str(row.get("problem_id")): row for row in raw_rows}
        if len(raw_rows) != 500 or len(raw_by_id) != 500:
            raise RuntimeError(f"native raw generations are incomplete: {variant}")
        for shard_index, shard_file in enumerate(shard_files):
            shard_rows: list[dict[str, Any]] = []
            for benchmark_row in native.controller.read_jsonl(shard_file):
                problem_id = str(benchmark_row["problem_id"])
                if problem_id not in benchmark_by_id or problem_id not in raw_by_id:
                    raise RuntimeError(f"missing native join key: {variant}/{problem_id}")
                raw = raw_by_id[problem_id]
                if str(raw.get("variant")) != variant:
                    raise RuntimeError(f"raw generation variant mismatch: {variant}/{problem_id}")
                response = raw.get("generation")
                if not isinstance(response, str):
                    raise RuntimeError(f"raw generation is not text: {variant}/{problem_id}")
                gate_row = {
                    "problem": str(benchmark_row["problem"]),
                    "problem_id": problem_id,
                    "response": response,
                    "response_sha256": sha256_text(response),
                    "shard_index": shard_index,
                    "variant": variant,
                }
                if set(gate_row) != ALLOWED_GATE_INPUT_FIELDS:
                    raise AssertionError("gate redaction field allow-list changed")
                shard_rows.append(gate_row)
                input_records.append(gate_row)
            if len(shard_rows) != ROWS_PER_SHARD:
                raise RuntimeError(f"gate shard is not 125 rows: {variant}/{shard_index}")
            native.controller.atomic_text(
                input_path(repo_root, variant, shard_index),
                native.controller.jsonl_text(shard_rows),
            )
    native.controller.atomic_text(paths["gold"], native.controller.jsonl_text(gold_records))
    canary_rows = select_canary_rows(repo_root, input_records)
    native.controller.atomic_text(
        paths["canary_inputs"], native.controller.jsonl_text(canary_rows)
    )
    metadata = {
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "canary_rows": len(canary_rows),
        "created_at": native.controller.now_iso(),
        "dataset_jsonl_sha256": native.controller.sha256_file(benchmark),
        "gate_input_fields": sorted(ALLOWED_GATE_INPUT_FIELDS),
        "gold_location": "intentionally omitted from no-gold gate metadata",
        "policy_sha256": policy_sha,
        "policy_manifest": gate_policy_manifest(),
        "rows": len(input_records),
        "shards": EXPECTED_VARIANTS * EXPECTED_SHARDS,
        "variants": variants(value),
    }
    native.controller.atomic_text(paths["gate_metadata"], native.controller.canonical_json(metadata))
    print(
        f"LLAMA_GATE_INPUTS_PREPARED rows={len(input_records)} shards=48 "
        f"canary_rows={len(canary_rows)} root={paths['gate']}",
        flush=True,
    )


def trim_span(response: str, start: int, end: int) -> tuple[int, int]:
    while start < end and response[start].isspace():
        start += 1
    while end > start and response[end - 1].isspace():
        end -= 1
    return start, end


def balanced_box_spans(response: str) -> list[dict[str, int]]:
    """Locate exact ``\boxed{...}`` wrappers and their content spans."""
    spans: list[dict[str, int]] = []
    for match in re.finditer(r"\\boxed\s*\{", response):
        opening = response.find("{", match.start(), match.end())
        depth = 0
        index = opening
        while index < len(response):
            character = response[index]
            escaped = index > 0 and response[index - 1] == "\\"
            if not escaped and character == "{":
                depth += 1
            elif not escaped and character == "}":
                depth -= 1
                if depth == 0:
                    spans.append(
                        {
                            "wrapper_start": match.start(),
                            "content_start": opening + 1,
                            "content_end": index,
                            "wrapper_end": index + 1,
                        }
                    )
                    break
            index += 1
    return spans


def response_regions(response: str) -> list[dict[str, Any]]:
    """Build a bounded, deterministic evidence catalog without changing text."""
    candidates: dict[tuple[int, int], set[str]] = {}

    def add(start: int, end: int, kind: str) -> None:
        start, end = trim_span(response, start, end)
        if start < end:
            candidates.setdefault((start, end), set()).add(kind)

    lines = list(re.finditer(r"[^\n]+", response))
    for line in lines[-24:]:
        add(line.start(), line.end(), "terminal_line")
    for line in lines:
        text = line.group(0)
        if r"\boxed" in text:
            add(line.start(), line.end(), "boxed_line")
        if re.search(
            r"\b(?:final answer|answer is|therefore|thus|hence|correction|however|instead)\b",
            text,
            re.I,
        ):
            add(line.start(), line.end(), "answer_state_line")
    paragraphs = list(re.finditer(r"(?:^|\n\s*\n)(.*?)(?=\n\s*\n|\Z)", response, re.S))
    for paragraph in paragraphs[-8:]:
        add(paragraph.start(1), paragraph.end(1), "terminal_paragraph")
    for box in balanced_box_spans(response):
        add(box["wrapper_start"], box["wrapper_end"], "boxed_expression")
        line_start = response.rfind("\n", 0, box["wrapper_start"]) + 1
        line_end = response.find("\n", box["wrapper_end"])
        add(line_start, len(response) if line_end < 0 else line_end, "boxed_line")
    if len(response) <= 1200:
        add(0, len(response), "full_response")
    ordered = sorted(candidates.items(), key=lambda item: (item[0][0], item[0][1]))
    return [
        {
            "end": end,
            "id": f"R{index:03d}",
            "kind": "+".join(sorted(kinds)),
            "start": start,
            "text": response[start:end],
        }
        for index, ((start, end), kinds) in enumerate(ordered)
    ]


def answer_candidates(response: str, region: dict[str, Any]) -> list[dict[str, Any]]:
    """Offer common exact spans; lexer boundaries remain the exhaustive fallback."""
    region_start, region_end = int(region["start"]), int(region["end"])
    spans: dict[tuple[int, int], dict[str, Any]] = {}
    priorities = {
        "boxed_content": 0,
        "math_delimiter": 1,
        "answer_cue_tail": 2,
        "balanced_group": 3,
        "mcq_or_scalar": 4,
        "evidence_region": 5,
    }

    def add(
        start: int, end: int, origin: str, boxed: bool = False, trim: bool = True
    ) -> None:
        if trim:
            start, end = trim_span(response, start, end)
        if not (region_start <= start < end <= region_end):
            return
        key = (start, end)
        current = spans.get(key)
        value = {
            "boxed": boxed,
            "end": end,
            "origin": origin,
            "start": start,
        }
        if current is None or priorities[origin] < priorities[current["origin"]]:
            spans[key] = value

    boxes = balanced_box_spans(response)
    for box in boxes:
        if region_start <= box["content_start"] and box["content_end"] <= region_end:
            add(
                box["content_start"],
                box["content_end"],
                "boxed_content",
                True,
                trim=False,
            )

    region_text = response[region_start:region_end]
    delimiter_patterns = (
        re.compile(r"\$\$(.+?)\$\$", re.S),
        re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.S),
        re.compile(r"\\\[(.+?)\\\]", re.S),
        re.compile(r"\\\((.+?)\\\)", re.S),
    )
    for pattern in delimiter_patterns:
        for match in pattern.finditer(region_text):
            add(region_start + match.start(), region_start + match.end(), "math_delimiter")

    cue = re.compile(
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)$", re.I | re.S
    )
    cue_match = cue.search(region_text)
    if cue_match:
        start, end = region_start + cue_match.start(1), region_start + cue_match.end(1)
        start, end = trim_span(response, start, end)
        if end > start and response[end - 1] in ".;" and not (
            end - start >= 2 and response[end - 2].isdigit() and response[end - 1] == "."
        ):
            end -= 1
        add(start, end, "answer_cue_tail")

    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for offset, character in enumerate(region_text):
        absolute = region_start + offset
        escaped = absolute > 0 and response[absolute - 1] == "\\"
        if not escaped and character in "([{":
            stack.append((character, absolute))
        elif not escaped and character in pairs and stack and stack[-1][0] == pairs[character]:
            _, start = stack.pop()
            if "," in response[start : absolute + 1] or "=" in response[start : absolute + 1]:
                add(start, absolute + 1, "balanced_group")

    token_pattern = re.compile(
        r"\\text\s*\{\s*\([A-Ea-e]\)\s*\}|\([A-Ea-e]\)|"
        r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?(?![\w.])"
    )
    for match in token_pattern.finditer(region_text):
        add(region_start + match.start(), region_start + match.end(), "mcq_or_scalar")
    add(region_start, region_end, "evidence_region")

    ordered = sorted(
        spans.values(),
        key=lambda value: (
            priorities[value["origin"]],
            value["start"],
            value["end"],
        ),
    )[:96]
    return [
        {**value, "id": f"C{index:03d}", "text": response[value["start"] : value["end"]]}
        for index, value in enumerate(ordered)
    ]


def lexer_boundaries(
    response: str, region: dict[str, Any], candidates: Sequence[dict[str, Any]]
) -> list[dict[str, int | str]]:
    """Expose every LaTeX-aware token boundary as an ID, never as generated text."""
    start, end = int(region["start"]), int(region["end"])
    positions = {start, end}
    token_pattern = re.compile(r"\\[A-Za-z]+|\\.|[A-Za-z]+|\d+(?:\.\d+)?|\s+|.", re.S)
    for token in token_pattern.finditer(response, start, end):
        positions.update((token.start(), token.end()))
    for candidate in candidates:
        positions.update((int(candidate["start"]), int(candidate["end"])))
    for box in balanced_box_spans(response):
        if start <= box["content_start"] <= box["content_end"] <= end:
            positions.update((box["content_start"], box["content_end"]))
    return [
        {"id": f"B{index:03d}", "position": position}
        for index, position in enumerate(sorted(positions))
    ]


def adjudication_schema(regions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    region_ids = [region["id"] for region in regions]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(ADJUDICATION_FIELDS),
        "properties": {
            "definitive": {"type": "boolean"},
            "boxed_final_answer": {"type": "boolean"},
            "format_compliant": {"type": "boolean"},
            "answer_form": {"type": "string", "enum": sorted(ANSWER_FORMS)},
            "evidence_region_id": {
                "type": ["string", "null"],
                "enum": [None, *region_ids],
            },
            "reason_code": {
                "type": "string",
                "enum": sorted(COMMITTED_REASONS | NONCOMMITTED_REASONS),
            },
        },
    }


def extraction_schema(
    candidates: Sequence[dict[str, Any]], boundaries: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(EXTRACTION_FIELDS),
        "properties": {
            "extraction_mode": {"type": "string", "enum": ["candidate", "boundary"]},
            "candidate_id": {
                "type": ["string", "null"],
                "enum": [None, *[candidate["id"] for candidate in candidates]],
            },
            "start_boundary_id": {
                "type": ["string", "null"],
                "enum": [None, *[boundary["id"] for boundary in boundaries]],
            },
            "end_boundary_id": {
                "type": ["string", "null"],
                "enum": [None, *[boundary["id"] for boundary in boundaries]],
            },
        },
    }


def validate_adjudication(
    value: Any, regions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ADJUDICATION_FIELDS:
        raise RuntimeError("adjudication does not have the exact schema")
    for field in ("definitive", "boxed_final_answer", "format_compliant"):
        if type(value[field]) is not bool:
            raise RuntimeError(f"{field} is not boolean")
    if value["answer_form"] not in ANSWER_FORMS:
        raise RuntimeError("unknown answer_form")
    if value["reason_code"] not in COMMITTED_REASONS | NONCOMMITTED_REASONS:
        raise RuntimeError("unknown reason_code")
    region_ids = {region["id"] for region in regions}
    if value["evidence_region_id"] is not None and value["evidence_region_id"] not in region_ids:
        raise RuntimeError("unknown evidence_region_id")
    if value["format_compliant"] != value["boxed_final_answer"]:
        raise RuntimeError("format_compliant must equal boxed_final_answer")
    if value["definitive"]:
        if value["reason_code"] not in COMMITTED_REASONS:
            raise RuntimeError("definitive adjudication has a non-commitment reason")
        if value["answer_form"] == "none" or value["evidence_region_id"] is None:
            raise RuntimeError("definitive adjudication lacks answer form or evidence")
        if value["reason_code"] == "single_committed_unboxed_answer" and value["boxed_final_answer"]:
            raise RuntimeError("unboxed reason conflicts with boxing decision")
    else:
        if value["reason_code"] not in NONCOMMITTED_REASONS:
            raise RuntimeError("nondefinitive adjudication has a commitment reason")
        if value["answer_form"] != "none":
            raise RuntimeError("nondefinitive adjudication must use answer_form=none")
        if value["boxed_final_answer"] or value["format_compliant"]:
            raise RuntimeError("nondefinitive adjudication cannot be format compliant")
    return value


def selected_extraction_span(
    value: Any,
    response: str,
    adjudication: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    boundaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != EXTRACTION_FIELDS:
        raise RuntimeError("extraction does not have the exact schema")
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    boundary_by_id = {boundary["id"]: boundary for boundary in boundaries}
    if value["extraction_mode"] == "candidate":
        if value["candidate_id"] not in candidate_by_id:
            raise RuntimeError("candidate extraction lacks a known candidate_id")
        if value["start_boundary_id"] is not None or value["end_boundary_id"] is not None:
            raise RuntimeError("candidate extraction also selected boundaries")
        candidate = candidate_by_id[value["candidate_id"]]
        start, end = int(candidate["start"]), int(candidate["end"])
    elif value["extraction_mode"] == "boundary":
        if value["candidate_id"] is not None:
            raise RuntimeError("boundary extraction also selected a candidate")
        if value["start_boundary_id"] not in boundary_by_id or value["end_boundary_id"] not in boundary_by_id:
            raise RuntimeError("boundary extraction lacks known boundary IDs")
        start = int(boundary_by_id[value["start_boundary_id"]]["position"])
        end = int(boundary_by_id[value["end_boundary_id"]]["position"])
        if start >= end:
            raise RuntimeError("boundary extraction has an empty or reversed span")
    else:
        raise RuntimeError("unknown extraction_mode")

    boxes = balanced_box_spans(response)
    exact_box = next(
        (
            box
            for box in boxes
            if box["content_start"] == start and box["content_end"] == end
        ),
        None,
    )
    inside_box = any(
        box["content_start"] <= start < end <= box["content_end"] for box in boxes
    )
    if adjudication["boxed_final_answer"] and exact_box is None:
        raise RuntimeError("boxed adjudication did not select one complete boxed content span")
    if not adjudication["boxed_final_answer"] and inside_box:
        raise RuntimeError("unboxed adjudication selected text from inside a box")
    if not response[start:end].strip():
        raise RuntimeError("selected answer span is blank")
    return {
        **value,
        "boxed_content": exact_box is not None,
        "end": end,
        "start": start,
    }


def exact_excerpt(response: str, start: int, end: int) -> str:
    line_start = response.rfind("\n", 0, start) + 1
    line_end = response.find("\n", end)
    if line_end < 0:
        line_end = len(response)
    line_start, line_end = trim_span(response, line_start, line_end)
    if 0 < line_end - line_start <= 320:
        return response[line_start:line_end]
    return response[start:end]


def terminal_decision(
    response: str,
    adjudication: dict[str, Any],
    regions: Sequence[dict[str, Any]],
    extraction: dict[str, Any] | None,
) -> dict[str, Any]:
    if not adjudication["definitive"]:
        region_by_id = {region["id"]: region for region in regions}
        evidence = region_by_id.get(adjudication["evidence_region_id"])
        conflict_quote = None
        if evidence is not None:
            conflict_quote = exact_excerpt(
                response, int(evidence["start"]), int(evidence["end"])
            )
        return {
            "definitive": False,
            "final_answer_raw": None,
            "boxed_final_answer": False,
            "format_compliant": False,
            "answer_form": "none",
            "supporting_quote": None,
            "later_conflict_quote": conflict_quote,
            "reason_code": adjudication["reason_code"],
        }
    if extraction is None:
        raise RuntimeError("definitive adjudication lacks extraction")
    start, end = int(extraction["start"]), int(extraction["end"])
    return {
        "definitive": True,
        "final_answer_raw": response[start:end],
        "boxed_final_answer": bool(adjudication["boxed_final_answer"]),
        "format_compliant": bool(adjudication["format_compliant"]),
        "answer_form": adjudication["answer_form"],
        "supporting_quote": exact_excerpt(response, start, end),
        "later_conflict_quote": None,
        "reason_code": adjudication["reason_code"],
    }


def validate_decision(value: Any, response: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(TERMINAL_DECISION_SCHEMA["required"]):
        raise RuntimeError("gate output does not have the exact decision schema")
    if type(value["definitive"]) is not bool:
        raise RuntimeError("definitive is not boolean")
    for field in ("boxed_final_answer", "format_compliant"):
        if type(value[field]) is not bool:
            raise RuntimeError(f"{field} is not boolean")
    if value["answer_form"] not in ANSWER_FORMS:
        raise RuntimeError("unknown answer_form")
    if value["reason_code"] not in COMMITTED_REASONS | NONCOMMITTED_REASONS:
        raise RuntimeError("unknown reason_code")
    for field in ("final_answer_raw", "supporting_quote", "later_conflict_quote"):
        if value[field] is not None and not isinstance(value[field], str):
            raise RuntimeError(f"{field} must be a string or null")
    if value["format_compliant"] != value["boxed_final_answer"]:
        raise RuntimeError("format_compliant must equal boxed_final_answer")
    if value["definitive"]:
        if value["reason_code"] not in COMMITTED_REASONS:
            raise RuntimeError("definitive decision has a non-commitment reason")
        if (
            not isinstance(value["final_answer_raw"], str)
            or not value["final_answer_raw"].strip()
            or not isinstance(value["supporting_quote"], str)
            or not value["supporting_quote"].strip()
        ):
            raise RuntimeError("definitive decision lacks answer or supporting quote")
        if value["answer_form"] == "none" or value["later_conflict_quote"] is not None:
            raise RuntimeError("definitive decision has incompatible fields")
        if value["final_answer_raw"] not in response:
            raise RuntimeError("final_answer_raw is not copied exactly from the response")
        if value["supporting_quote"] not in response:
            raise RuntimeError("supporting_quote is not an exact response excerpt")
        if (
            value["reason_code"] == "single_committed_unboxed_answer"
            and value["boxed_final_answer"]
        ):
            raise RuntimeError("unboxed reason conflicts with boxing decision")
    else:
        if value["reason_code"] not in NONCOMMITTED_REASONS:
            raise RuntimeError("nondefinitive decision has a commitment reason")
        if value["final_answer_raw"] is not None or value["supporting_quote"] is not None:
            raise RuntimeError("nondefinitive decision extracted an answer")
        if value["answer_form"] != "none":
            raise RuntimeError("nondefinitive decision must use answer_form=none")
        if value["boxed_final_answer"] or value["format_compliant"]:
            raise RuntimeError("nondefinitive decision cannot be format compliant")
        if value["later_conflict_quote"] is not None and value["later_conflict_quote"] not in response:
            raise RuntimeError("later_conflict_quote is not an exact response excerpt")
    return value


def validated_gate_inputs(
    repo_root: Path, variant: str, shard_index: int
) -> tuple[list[dict[str, Any]], str]:
    value = inventory(repo_root)
    if variant not in variants(value) or shard_index not in range(EXPECTED_SHARDS):
        raise RuntimeError(f"unknown gate shard: {variant}/{shard_index}")
    metadata = native.controller.load_json(roots(repo_root)["gate_metadata"])
    if metadata.get("policy_sha256") != gate_policy_sha256():
        raise RuntimeError("prepared gate policy identity differs from the running code")
    if metadata.get("gate_input_fields") != sorted(ALLOWED_GATE_INPUT_FIELDS):
        raise RuntimeError("gate input metadata field allow-list mismatch")
    rows = native.controller.read_jsonl(input_path(repo_root, variant, shard_index))
    if len(rows) != ROWS_PER_SHARD:
        raise RuntimeError("gate input shard must contain exactly 125 rows")
    identifiers: set[str] = set()
    for row in rows:
        if set(row) != ALLOWED_GATE_INPUT_FIELDS:
            raise RuntimeError("gate input contains a forbidden field")
        if row["variant"] != variant or int(row["shard_index"]) != shard_index:
            raise RuntimeError("gate input shard identity mismatch")
        if sha256_text(str(row["response"])) != row["response_sha256"]:
            raise RuntimeError("gate response hash mismatch")
        identifiers.add(str(row["problem_id"]))
    if len(identifiers) != ROWS_PER_SHARD:
        raise RuntimeError("gate input contains duplicate problem IDs")
    return rows, str(metadata["policy_sha256"])


def source_identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["variant"]), int(row["shard_index"]), str(row["problem_id"])


def validate_gate_journal_row(
    row: dict[str, Any], inputs: dict[str, dict[str, Any]], policy_sha: str
) -> None:
    required = {
        "adjudication",
        "attempts",
        "decision",
        "extraction",
        "gate_status",
        "model_revision",
        "policy_sha256",
        "problem_id",
        "response_sha256",
        "row_key",
        "shard_index",
        "unresolved",
        "variant",
    }
    if set(row) != required:
        raise RuntimeError("gate journal row has unexpected fields")
    problem_id = str(row["problem_id"])
    source = inputs.get("\0".join(map(str, source_identity(row))))
    if source is None:
        raise RuntimeError("gate journal row is not in the input shard")
    expected_key = stable_row_key(
        str(source["variant"]), problem_id, str(source["response_sha256"]), policy_sha
    )
    if row["row_key"] != expected_key:
        raise RuntimeError("gate journal row key mismatch")
    if row["response_sha256"] != source["response_sha256"]:
        raise RuntimeError("gate journal response hash mismatch")
    if row["policy_sha256"] != policy_sha:
        raise RuntimeError("gate journal policy hash mismatch")
    if row["model_revision"] != policy()["model"]["revision"]:
        raise RuntimeError("gate journal model revision mismatch")
    if row["gate_status"] == "resolved":
        if row["unresolved"] is not None or row["decision"] is None:
            raise RuntimeError("resolved gate row has incompatible terminal fields")
        response = str(source["response"])
        decision = validate_decision(row["decision"], response)
        regions = response_regions(response)
        adjudication = validate_adjudication(row["adjudication"], regions)
        if decision["definitive"]:
            extraction = row["extraction"]
            if not isinstance(extraction, dict):
                raise RuntimeError("definitive gate row lacks extraction metadata")
            start, end = extraction.get("start"), extraction.get("end")
            if type(start) is not int or type(end) is not int or response[start:end] != decision["final_answer_raw"]:
                raise RuntimeError("stored answer is not the mechanically selected source span")
            if bool(extraction.get("boxed_content")) != decision["boxed_final_answer"]:
                raise RuntimeError("stored extraction boxing metadata disagrees with decision")
        elif row["extraction"] is not None or adjudication["definitive"]:
            raise RuntimeError("nondefinitive gate row contains extraction metadata")
    elif row["gate_status"] == "unresolved":
        if row["unresolved"] is None or row["decision"] is not None:
            raise RuntimeError("unresolved gate row has incompatible terminal fields")
    else:
        raise RuntimeError("unknown gate_status")


def repair_gate_journal(
    output: Path, source_rows: list[dict[str, Any]], policy_sha: str
) -> dict[str, dict[str, Any]]:
    """Drop a torn suffix and deduplicate valid rows in canonical input order."""
    inputs = {"\0".join(map(str, source_identity(row))): row for row in source_rows}
    candidates: dict[str, dict[str, Any]] = {}
    if output.is_file():
        with output.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    break
                try:
                    row = json.loads(line)
                    validate_gate_journal_row(row, inputs, policy_sha)
                except (json.JSONDecodeError, RuntimeError):
                    if line_number == sum(1 for _ in output.open(encoding="utf-8")):
                        break
                    raise
                candidates["\0".join(map(str, source_identity(row)))] = row
    ordered = [
        candidates["\0".join(map(str, source_identity(source)))]
        for source in source_rows
        if "\0".join(map(str, source_identity(source))) in candidates
    ]
    if output.exists() or ordered:
        native.controller.atomic_text(output, native.controller.jsonl_text(ordered))
    return {"\0".join(map(str, source_identity(row))): row for row in ordered}


def append_fsynced(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def render_adjudication_prompt(
    template: str, problem: str, response: str, regions: Sequence[dict[str, Any]]
) -> str:
    replacements = {
        "{{PROBLEM}}": problem,
        "{{MODEL_RESPONSE}}": response,
        "{{EVIDENCE_REGIONS}}": "\n".join(
            f"{region['id']} ({region['kind']}): "
            f"{json.dumps(region['text'], ensure_ascii=False)}"
            for region in regions
        ),
    }
    for placeholder, value in replacements.items():
        if template.count(placeholder) != 1:
            raise RuntimeError(f"adjudication prompt placeholder changed: {placeholder}")
        template = template.replace(placeholder, value)
    return template


def render_extraction_prompt(
    template: str,
    problem: str,
    adjudication: dict[str, Any],
    response: str,
    region: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    boundaries: Sequence[dict[str, Any]],
) -> str:
    boundary_chunks: list[str] = []
    for index, boundary in enumerate(boundaries):
        boundary_chunks.append(f"<{boundary['id']}>")
        if index + 1 < len(boundaries):
            left = int(boundary["position"])
            right = int(boundaries[index + 1]["position"])
            boundary_chunks.append(response[left:right])
    replacements = {
        "{{PROBLEM}}": problem,
        "{{ADJUDICATION}}": json.dumps(adjudication, ensure_ascii=False, sort_keys=True),
        "{{EVIDENCE}}": json.dumps(region["text"], ensure_ascii=False),
        "{{CANDIDATES}}": "\n".join(
            f"{candidate['id']} ({candidate['origin']}, boxed={str(candidate['boxed']).lower()}): "
            f"{json.dumps(candidate['text'], ensure_ascii=False)}"
            for candidate in candidates
        ),
        "{{BOUNDARIES}}": "".join(boundary_chunks),
    }
    for placeholder, value in replacements.items():
        if template.count(placeholder) != 1:
            raise RuntimeError(f"extraction prompt placeholder changed: {placeholder}")
        template = template.replace(placeholder, value)
    return template


def isolate_vllm_runtime_env(repo_root: Path, variant: str, shard_index: int) -> None:
    """Keep concurrent/requeued vLLM workers from racing on compiled caches."""
    task_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or variant
    cache_key = f"{task_id}_{shard_index}"
    default_root = repo_root / "checkpoints" / "cache" / "llama-gate"
    defaults = {
        "VLLM_CACHE_ROOT": default_root / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": default_root / "torchinductor",
        "TRITON_CACHE_DIR": default_root / "triton",
    }
    for name, default in defaults.items():
        current = Path(os.environ.get(name, str(default)))
        isolated = current if current.name == cache_key else current / cache_key
        isolated.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(isolated)
    os.environ["VLLM_NO_USAGE_STATS"] = "1"


def execute_gate_rows(
    repo_root: Path,
    source_rows: Sequence[dict[str, Any]],
    output: Path,
    failure_output: Path,
    policy_sha: str,
    identity: str,
) -> dict[str, dict[str, Any]]:
    """Run both constrained-ID passes and persist one terminal row per input."""
    paths = roots(repo_root)
    gate_policy = policy()
    completed = repair_gate_journal(output, list(source_rows), policy_sha)
    pending = [
        row
        for row in source_rows
        if "\0".join(map(str, source_identity(row))) not in completed
    ]
    if not pending:
        return completed

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    ready = validate_model(repo_root)
    if ready.get("revision") != gate_policy["model"]["revision"]:
        raise RuntimeError("local Llama model is not the pinned revision")
    adjudication_template = (
        HANDOFF / gate_policy["gate"]["adjudication_prompt"]
    ).read_text(encoding="utf-8")
    extraction_template = (
        HANDOFF / gate_policy["gate"]["extraction_prompt"]
    ).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(str(paths["model"]), local_files_only=True)
    max_length = int(gate_policy["engine"]["max_model_len"])
    adjudication_tokens = int(gate_policy["gate"]["adjudication_max_tokens"])
    extraction_tokens = int(gate_policy["gate"]["extraction_max_tokens"])

    rendered: list[dict[str, Any]] = []
    oversized: list[dict[str, Any]] = []
    for row in pending:
        regions = response_regions(str(row["response"]))
        content = render_adjudication_prompt(
            adjudication_template,
            str(row["problem"]),
            str(row["response"]),
            regions,
        )
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_count = len(tokenizer(chat, add_special_tokens=False)["input_ids"])
        if token_count + adjudication_tokens > max_length:
            oversized.append(
                {
                    "error": (
                        "adjudication exceeds pinned context without truncation: "
                        f"tokens={token_count}+{adjudication_tokens}"
                    ),
                    "regions": regions,
                    "row": row,
                }
            )
            continue
        rendered.append(
            {
                "chat": chat,
                "content": content,
                "regions": regions,
                "row": row,
                "schema": adjudication_schema(regions),
            }
        )

    engine = gate_policy["engine"]
    llm = LLM(
        model=str(paths["model"]),
        tensor_parallel_size=int(engine["tensor_parallel_size"]),
        quantization=str(engine["quantization"]),
        kv_cache_dtype=str(engine["kv_cache_dtype"]),
        dtype=str(engine["dtype"]),
        max_model_len=max_length,
        max_num_seqs=int(engine["max_num_seqs"]),
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        enforce_eager=bool(engine["enforce_eager"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        enable_chunked_prefill=bool(engine["enable_chunked_prefill"]),
        calculate_kv_scales=bool(engine["calculate_kv_scales"]),
        trust_remote_code=False,
    )

    def sampling(schema: dict[str, Any], max_tokens: int) -> Any:
        return SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            seed=0,
            guided_decoding=GuidedDecodingParams(json=schema),
        )

    def batch_decode(
        prompts: Sequence[str], schemas: Sequence[dict[str, Any]], max_tokens: int, tqdm: bool
    ) -> list[str]:
        outputs = llm.generate(
            list(prompts),
            [sampling(schema, max_tokens) for schema in schemas],
            use_tqdm=tqdm,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned the wrong constrained batch size")
        decoded: list[str] = []
        for request_output in outputs:
            if len(request_output.outputs) != 1:
                raise RuntimeError("constrained request did not produce exactly one completion")
            decoded.append(request_output.outputs[0].text)
        return decoded

    semantic_retries = int(gate_policy["gate"]["semantic_retries"])

    def validate_with_retries(
        *,
        stage: str,
        item: dict[str, Any],
        decoded: str,
        schema: dict[str, Any],
        max_tokens: int,
        validator: Any,
    ) -> tuple[Any | None, int, str | None]:
        last_error: str | None = None
        for attempt in range(semantic_retries + 1):
            try:
                return validator(json.loads(decoded)), attempt + 1, None
            except (json.JSONDecodeError, RuntimeError) as error:
                last_error = str(error)
                row = item["row"]
                append_fsynced(
                    failure_output,
                    {
                        "attempt": attempt + 1,
                        "decoded": decoded,
                        "error": last_error,
                        "problem_id": str(row["problem_id"]),
                        "response_sha256": str(row["response_sha256"]),
                        "stage": stage,
                        "variant": str(row["variant"]),
                    },
                )
                if attempt == semantic_retries:
                    return None, attempt + 1, last_error
                feedback = (
                    f"The previous {stage} JSON failed mechanical validation: {last_error}. "
                    "Return corrected JSON only. Keep the substantive adjudication unchanged. "
                    "Select only IDs listed in the original request; never emit answer text."
                )
                retry_chat = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": item["content"]},
                        {"role": "assistant", "content": decoded},
                        {"role": "user", "content": feedback},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                retry_count = len(
                    tokenizer(retry_chat, add_special_tokens=False)["input_ids"]
                )
                if retry_count + max_tokens > max_length:
                    return None, attempt + 1, "semantic retry exceeds pinned context"
                decoded = batch_decode([retry_chat], [schema], max_tokens, False)[0]
        raise AssertionError("unreachable retry loop")

    def journal_row(
        item: dict[str, Any],
        adjudication: dict[str, Any] | None,
        extraction: dict[str, Any] | None,
        decision: dict[str, Any] | None,
        attempts: dict[str, int],
        unresolved: dict[str, Any] | None,
    ) -> dict[str, Any]:
        row = item["row"]
        return {
            "adjudication": adjudication,
            "attempts": attempts,
            "decision": decision,
            "extraction": extraction,
            "gate_status": "unresolved" if unresolved is not None else "resolved",
            "model_revision": gate_policy["model"]["revision"],
            "policy_sha256": policy_sha,
            "problem_id": str(row["problem_id"]),
            "response_sha256": str(row["response_sha256"]),
            "row_key": stable_row_key(
                str(row["variant"]),
                str(row["problem_id"]),
                str(row["response_sha256"]),
                policy_sha,
            ),
            "shard_index": int(row["shard_index"]),
            "unresolved": unresolved,
            "variant": str(row["variant"]),
        }

    chunk_size = int(gate_policy["gate"]["chunk_size"])
    persisted = 0
    for item in oversized:
        append_fsynced(
            output,
            journal_row(
                item,
                None,
                None,
                None,
                {"adjudication": 0, "extraction": 0},
                {"attempts": 0, "last_error": item["error"], "stage": "adjudication"},
            ),
        )
        persisted += 1
    for start in range(0, len(rendered), chunk_size):
        batch = rendered[start : start + chunk_size]
        initial = batch_decode(
            [item["chat"] for item in batch],
            [item["schema"] for item in batch],
            adjudication_tokens,
            True,
        )
        terminal: dict[str, dict[str, Any]] = {}
        definitive_items: list[dict[str, Any]] = []
        for item, decoded in zip(batch, initial, strict=True):
            adjudication, attempt_count, error = validate_with_retries(
                stage="adjudication",
                item=item,
                decoded=decoded,
                schema=item["schema"],
                max_tokens=adjudication_tokens,
                validator=lambda value, regions=item["regions"]: validate_adjudication(
                    value, regions
                ),
            )
            item["adjudication_attempts"] = attempt_count
            key = "\0".join(map(str, source_identity(item["row"])))
            if adjudication is None:
                terminal[key] = journal_row(
                    item,
                    None,
                    None,
                    None,
                    {"adjudication": attempt_count, "extraction": 0},
                    {"attempts": attempt_count, "last_error": error, "stage": "adjudication"},
                )
            elif not adjudication["definitive"]:
                decision = terminal_decision(
                    str(item["row"]["response"]), adjudication, item["regions"], None
                )
                terminal[key] = journal_row(
                    item,
                    adjudication,
                    None,
                    validate_decision(decision, str(item["row"]["response"])),
                    {"adjudication": attempt_count, "extraction": 0},
                    None,
                )
            else:
                region_by_id = {region["id"]: region for region in item["regions"]}
                region = region_by_id[adjudication["evidence_region_id"]]
                response = str(item["row"]["response"])
                candidates = answer_candidates(response, region)
                boundaries = lexer_boundaries(response, region, candidates)
                content = render_extraction_prompt(
                    extraction_template,
                    str(item["row"]["problem"]),
                    adjudication,
                    response,
                    region,
                    candidates,
                    boundaries,
                )
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if len(tokenizer(chat, add_special_tokens=False)["input_ids"]) + extraction_tokens > max_length:
                    terminal[key] = journal_row(
                        item,
                        adjudication,
                        None,
                        None,
                        {"adjudication": attempt_count, "extraction": 0},
                        {"attempts": 0, "last_error": "extraction prompt exceeds pinned context", "stage": "extraction"},
                    )
                    continue
                item.update(
                    {
                        "adjudication": adjudication,
                        "boundaries": boundaries,
                        "candidates": candidates,
                        "chat": chat,
                        "content": content,
                        "schema": extraction_schema(candidates, boundaries),
                    }
                )
                definitive_items.append(item)

        if definitive_items:
            selected = batch_decode(
                [item["chat"] for item in definitive_items],
                [item["schema"] for item in definitive_items],
                extraction_tokens,
                False,
            )
            for item, decoded in zip(definitive_items, selected, strict=True):
                response = str(item["row"]["response"])
                extraction, attempt_count, error = validate_with_retries(
                    stage="extraction",
                    item=item,
                    decoded=decoded,
                    schema=item["schema"],
                    max_tokens=extraction_tokens,
                    validator=lambda value, current=item: selected_extraction_span(
                        value,
                        str(current["row"]["response"]),
                        current["adjudication"],
                        current["candidates"],
                        current["boundaries"],
                    ),
                )
                key = "\0".join(map(str, source_identity(item["row"])))
                attempts = {
                    "adjudication": int(item["adjudication_attempts"]),
                    "extraction": attempt_count,
                }
                if extraction is None:
                    terminal[key] = journal_row(
                        item,
                        item["adjudication"],
                        None,
                        None,
                        attempts,
                        {"attempts": attempt_count, "last_error": error, "stage": "extraction"},
                    )
                else:
                    decision = terminal_decision(
                        response, item["adjudication"], item["regions"], extraction
                    )
                    terminal[key] = journal_row(
                        item,
                        item["adjudication"],
                        extraction,
                        validate_decision(decision, response),
                        attempts,
                        None,
                    )

        for item in batch:
            key = "\0".join(map(str, source_identity(item["row"])))
            append_fsynced(output, terminal[key])
            persisted += 1
            if not (paths["gate"] / "GATE_STARTED.json").exists():
                native.controller.atomic_text(
                    paths["gate"] / "GATE_STARTED.json",
                    native.controller.canonical_json(
                        {
                            "first_problem_id": item["row"]["problem_id"],
                            "first_shard": item["row"]["shard_index"],
                            "first_variant": item["row"]["variant"],
                            "identity": identity,
                            "started_at": native.controller.now_iso(),
                        }
                    ),
                )
        if stop_requested:
            print(f"LLAMA_GATE_REQUEUE_SAFE identity={identity} persisted={persisted}", flush=True)
            raise SystemExit(75)
    return repair_gate_journal(output, list(source_rows), policy_sha)


def run_gate(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    source_rows, policy_sha = validated_gate_inputs(repo_root, args.variant, args.shard_index)
    paths = roots(repo_root)
    isolate_vllm_runtime_env(repo_root, args.variant, args.shard_index)
    complete_path = decision_complete_path(repo_root, args.variant, args.shard_index)
    if args.overlay_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.overlay_id):
            raise RuntimeError("overlay ID must be a short filesystem-safe identifier")
        if not complete_path.is_file():
            raise RuntimeError("an unresolved-only overlay requires a complete base shard")
        base = repair_gate_journal(
            decision_path(repo_root, args.variant, args.shard_index), source_rows, policy_sha
        )
        source_rows = [
            row
            for row in source_rows
            if base["\0".join(map(str, source_identity(row)))]["gate_status"] == "unresolved"
        ]
        overlay_root = paths["gate_overlays"] / args.overlay_id / args.variant
        output = overlay_root / f"shard_{args.shard_index:02d}.jsonl"
        complete_path = overlay_root / f"shard_{args.shard_index:02d}.complete.json"
    else:
        output = decision_path(repo_root, args.variant, args.shard_index)
    repaired = execute_gate_rows(
        repo_root,
        source_rows,
        output,
        (
            paths["gate_outputs"] / "_validation_failures" / args.variant
            if args.overlay_id is None
            else paths["gate_overlays"] / args.overlay_id / "_validation_failures" / args.variant
        )
        / f"shard_{args.shard_index:02d}.jsonl",
        policy_sha,
        f"{args.variant}/{args.shard_index}",
    )
    expected_rows = len(source_rows)
    if len(repaired) != expected_rows:
        raise RuntimeError("gate shard ended without one durable terminal row per input")
    unresolved = sum(row["gate_status"] == "unresolved" for row in repaired.values())
    native.controller.atomic_text(
        complete_path,
        native.controller.canonical_json(
            {
                "policy_sha256": policy_sha,
                "overlay_id": args.overlay_id,
                "rows": expected_rows,
                "status": "complete",
                "unresolved": unresolved,
            }
        ),
    )
    print(
        f"LLAMA_GATE_SHARD_COMPLETE variant={args.variant} shard={args.shard_index} "
        f"overlay={args.overlay_id or 'base'} rows={expected_rows} unresolved={unresolved}",
        flush=True,
    )


def run_canary(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    paths = roots(repo_root)
    metadata = native.controller.load_json(paths["gate_metadata"])
    if metadata.get("policy_sha256") != gate_policy_sha256():
        raise RuntimeError("canary policy identity differs from prepared inputs")
    rows = native.controller.read_jsonl(paths["canary_inputs"])
    if not rows or len(rows) != metadata.get("canary_rows"):
        raise RuntimeError("canary input manifest is missing or incomplete")
    for row in rows:
        if set(row) != ALLOWED_GATE_INPUT_FIELDS:
            raise RuntimeError("canary contains a forbidden field")
        if sha256_text(str(row["response"])) != row["response_sha256"]:
            raise RuntimeError("canary response hash mismatch")
    isolate_vllm_runtime_env(repo_root, "canary", 0)
    repaired = execute_gate_rows(
        repo_root,
        rows,
        paths["canary_outputs"],
        paths["gate"] / "canary_validation_failures.jsonl",
        str(metadata["policy_sha256"]),
        "canary",
    )
    if len(repaired) != len(rows):
        raise RuntimeError("canary did not persist one terminal row per input")
    unresolved = sum(row["gate_status"] == "unresolved" for row in repaired.values())
    native.controller.atomic_text(
        paths["canary_complete"],
        native.controller.canonical_json(
            {
                "policy_sha256": metadata["policy_sha256"],
                "rows": len(rows),
                "status": "complete",
                "unresolved": unresolved,
            }
        ),
    )
    print(f"LLAMA_GATE_CANARY_COMPLETE rows={len(rows)} unresolved={unresolved}", flush=True)


def _math_verify_api() -> tuple[Any, Any, Any, Any, Any]:
    from math_verify import parse, verify
    from math_verify.parser import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        StringExtractionConfig,
    )

    return parse, verify, ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig


def parse_math(value: str) -> list[Any]:
    parse, _, ExprExtractionConfig, LatexExtractionConfig, _ = _math_verify_api()
    parsed = parse(
        f"${value.strip()}$",
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=5,
        raise_on_error=True,
    )
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError(f"expected one parsed mathematical object, found {len(parsed)}")
    return parsed


def equivalent_math(gold: str, candidate: str) -> tuple[bool, bool]:
    """Return (both parsed exactly once, mathematically equivalent)."""
    try:
        gold_parsed = parse_math(gold)
        candidate_parsed = parse_math(candidate)
        _, verify, _, _, _ = _math_verify_api()
        return True, bool(verify(gold_parsed, candidate_parsed, timeout_seconds=5))
    except Exception:
        return False, False


def normalize_text(value: str) -> str:
    result = value.strip()
    text_match = re.fullmatch(r"\\text\s*\{(.*)\}", result)
    if text_match:
        result = text_match.group(1)
    result = result.strip().strip("$").strip()
    choice = re.fullmatch(r"\(([A-Ea-e])\)", result)
    if choice:
        result = choice.group(1)
    return result.lower()


def equivalent_text(gold: str, candidate: str) -> tuple[bool, bool]:
    parse, verify, _, _, StringExtractionConfig = _math_verify_api()
    normalized_gold = normalize_text(gold)
    config = StringExtractionConfig(
        strings=(normalized_gold,), try_extract_without_anchor=True, lowercase=True
    )
    try:
        gold_parsed = parse(
            normalize_text(gold),
            extraction_config=[config],
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=5,
            raise_on_error=True,
        )
        candidate_parsed = parse(
            normalize_text(candidate),
            extraction_config=[config],
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=5,
            raise_on_error=True,
        )
        if len(gold_parsed) != 1 or len(candidate_parsed) != 1:
            return False, False
        return True, bool(verify(gold_parsed, candidate_parsed, timeout_seconds=5))
    except Exception:
        return False, False


def match_ordered(gold: Sequence[str], candidate: Sequence[str]) -> tuple[bool, bool]:
    parsed = True
    equivalent = True
    for gold_part, candidate_part in zip(gold, candidate, strict=True):
        part_parsed, part_equivalent = equivalent_math(gold_part, candidate_part)
        parsed = parsed and part_parsed
        equivalent = equivalent and part_equivalent
    return parsed, equivalent


def match_unordered(gold: Sequence[str], candidate: Sequence[str]) -> tuple[bool, bool]:
    """Require an exact-cardinality one-to-one symbolic matching."""
    if len(gold) != len(candidate):
        return False, False
    matrix: list[list[tuple[bool, bool]]] = [
        [equivalent_math(gold_part, candidate_part) for candidate_part in candidate]
        for gold_part in gold
    ]
    parsed = all(any(cell[0] for cell in row) for row in matrix) and all(
        any(matrix[row][column][0] for row in range(len(gold)))
        for column in range(len(candidate))
    )
    if not parsed:
        return False, False

    def search(gold_index: int, used: set[int]) -> bool:
        if gold_index == len(gold):
            return True
        for candidate_index, (_, is_equivalent) in enumerate(matrix[gold_index]):
            if is_equivalent and candidate_index not in used:
                if search(gold_index + 1, used | {candidate_index}):
                    return True
        return False

    return True, search(0, set())


def verify_structured_answer(
    gold_row: dict[str, Any], final_answer_raw: str
) -> dict[str, bool]:
    kind = str(gold_row["gold_type"])
    structure_matches = True
    parse_succeeded = False
    equivalent = False
    if kind in {"text", "multiple_choice"}:
        parse_succeeded, equivalent = equivalent_text(
            str(gold_row["correct_answer"]), final_answer_raw
        )
    elif kind == "ordered_tuple":
        candidate = tuple_parts(final_answer_raw)
        gold = [str(part) for part in gold_row["parts"]]
        structure_matches = candidate is not None and len(candidate) == len(gold)
        if structure_matches and candidate is not None:
            parse_succeeded, equivalent = match_ordered(gold, candidate)
    elif kind == "finite_set":
        candidate = set_parts(final_answer_raw)
        gold = [str(part) for part in gold_row["parts"]]
        structure_matches = candidate is not None and len(candidate) == len(gold)
        if structure_matches and candidate is not None:
            parse_succeeded, equivalent = match_unordered(gold, candidate)
    elif kind == "interval":
        candidate = interval_parts(final_answer_raw)
        gold = list(gold_row["parts"])
        structure_matches = candidate is not None and len(candidate) == len(gold)
        if structure_matches and candidate is not None:
            boundaries_match = all(
                candidate_part["left_boundary"] == gold_part["left_boundary"]
                and candidate_part["right_boundary"] == gold_part["right_boundary"]
                for gold_part, candidate_part in zip(gold, candidate, strict=True)
            )
            structure_matches = boundaries_match
            if boundaries_match:
                endpoint_gold: list[str] = []
                endpoint_candidate: list[str] = []
                for gold_part, candidate_part in zip(gold, candidate, strict=True):
                    endpoint_gold.extend(
                        [str(gold_part["left_endpoint"]), str(gold_part["right_endpoint"])]
                    )
                    endpoint_candidate.extend(
                        [candidate_part["left_endpoint"], candidate_part["right_endpoint"]]
                    )
                parse_succeeded, equivalent = match_ordered(endpoint_gold, endpoint_candidate)
    elif kind == "equation":
        structure_matches = equation_parts(final_answer_raw) is not None
        if structure_matches:
            parse_succeeded, equivalent = equivalent_math(
                str(gold_row["correct_answer"]), final_answer_raw
            )
    elif kind == "math":
        structure_matches = (
            equation_parts(final_answer_raw) is None
            and interval_parts(final_answer_raw) is None
            and tuple_parts(final_answer_raw) is None
            and set_parts(final_answer_raw) is None
        )
        if structure_matches:
            parse_succeeded, equivalent = equivalent_math(
                str(gold_row["correct_answer"]), final_answer_raw
            )
    else:
        raise RuntimeError(f"unknown gold type: {kind}")
    return {
        "mathematically_equivalent": equivalent,
        "structure_matches": structure_matches,
        "symbolic_parse_succeeded": parse_succeeded,
    }


def verify_answer_with_routing(
    gold_row: dict[str, Any], final_answer_raw: str
) -> dict[str, Any]:
    """Use the heuristic route first, then require compatible fallbacks to agree."""
    routes = list(gold_row.get("route_candidates") or [])
    if not routes:
        routes = [{"gold_type": gold_row["gold_type"], "parts": gold_row["parts"]}]
    audit: list[dict[str, Any]] = []
    for route in routes:
        route_row = {
            "correct_answer": gold_row["correct_answer"],
            "gold_type": route["gold_type"],
            "parts": route["parts"],
        }
        result = verify_structured_answer(route_row, final_answer_raw)
        audit.append({"gold_type": route["gold_type"], **result})

    compatible = [entry for entry in audit if entry["structure_matches"]]
    parsed = [entry for entry in compatible if entry["symbolic_parse_succeeded"]]
    if not compatible:
        return {
            "mathematically_equivalent": False,
            "route_audit": audit,
            "route_used": routes[0]["gold_type"],
            "structure_matches": False,
            "symbolic_parse_succeeded": False,
            "verification_status": "resolved_structure_mismatch",
        }
    if len(compatible) != len(parsed):
        return {
            "mathematically_equivalent": None,
            "route_audit": audit,
            "route_used": None,
            "structure_matches": True,
            "symbolic_parse_succeeded": False,
            "verification_status": "unresolved_verification",
        }
    outcomes = {bool(entry["mathematically_equivalent"]) for entry in parsed}
    if len(outcomes) != 1:
        return {
            "mathematically_equivalent": None,
            "route_audit": audit,
            "route_used": None,
            "structure_matches": True,
            "symbolic_parse_succeeded": True,
            "verification_status": "unresolved_route_conflict",
        }
    return {
        "mathematically_equivalent": outcomes.pop(),
        "route_audit": audit,
        "route_used": [entry["gold_type"] for entry in parsed],
        "structure_matches": True,
        "symbolic_parse_succeeded": True,
        "verification_status": "resolved",
    }


def all_gate_rows(repo_root: Path) -> list[dict[str, Any]]:
    value = inventory(repo_root)
    rows: list[dict[str, Any]] = []
    for variant in variants(value):
        for shard_index in range(EXPECTED_SHARDS):
            complete = decision_complete_path(repo_root, variant, shard_index)
            if not complete.is_file():
                raise RuntimeError(f"gate shard has no completion marker: {variant}/{shard_index}")
            source, policy_sha = validated_gate_inputs(repo_root, variant, shard_index)
            journal = repair_gate_journal(
                decision_path(repo_root, variant, shard_index), source, policy_sha
            )
            if len(journal) != ROWS_PER_SHARD:
                raise RuntimeError(f"gate shard is incomplete: {variant}/{shard_index}")
            rows.extend(
                journal["\0".join(map(str, source_identity(row)))] for row in source
            )
    if len(rows) != EXPECTED_ROWS or len({row["row_key"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError("gate output is not exactly 6000 unique decisions")
    by_key = {str(row["row_key"]): row for row in rows}
    overlay_root = roots(repo_root)["gate_overlays"]
    if overlay_root.is_dir():
        for overlay_path in sorted(overlay_root.glob("*/*/shard_*.jsonl")):
            overlay_rows = native.controller.read_jsonl(overlay_path)
            for overlay in overlay_rows:
                key = str(overlay.get("row_key"))
                base = by_key.get(key)
                if base is None:
                    raise RuntimeError(f"overlay contains an unknown gate row: {overlay_path}")
                if base["gate_status"] != "unresolved":
                    raise RuntimeError(f"overlay tries to replace a resolved gate row: {overlay_path}")
                if overlay.get("policy_sha256") != base["policy_sha256"]:
                    raise RuntimeError(f"overlay policy differs from base gate: {overlay_path}")
                if overlay.get("gate_status") == "resolved":
                    by_key[key] = overlay
    return [by_key[str(row["row_key"])] for row in rows]


def score_row(
    gate_row: dict[str, Any],
    gold_row: dict[str, Any],
    native_row: dict[str, Any],
) -> dict[str, Any]:
    decision = gate_row["decision"]
    verification = {
        "mathematically_equivalent": None,
        "route_audit": [],
        "route_used": None,
        "structure_matches": None,
        "symbolic_parse_succeeded": None,
        "verification_status": "not_run",
    }
    if gate_row["gate_status"] == "unresolved":
        correct: bool | None = None
    elif not decision["definitive"]:
        correct = False
        verification["verification_status"] = "not_applicable_nondefinitive"
    else:
        verification = verify_answer_with_routing(
            gold_row, str(decision["final_answer_raw"])
        )
        if verification["verification_status"].startswith("unresolved"):
            correct = None
        else:
            correct = bool(
                verification["symbolic_parse_succeeded"]
                and verification["structure_matches"]
                and verification["mathematically_equivalent"]
            )
    gate_record_sha = sha256_text(native.controller.canonical_json(gate_row))
    return {
        "answer_form": decision["answer_form"] if decision is not None else None,
        "boxed_final_answer": (
            bool(decision["boxed_final_answer"]) if decision is not None else None
        ),
        "correct": correct,
        "definitive": bool(decision["definitive"]) if decision is not None else None,
        "final_answer_raw": decision["final_answer_raw"] if decision is not None else None,
        "format_compliant": (
            bool(decision["format_compliant"]) if decision is not None else None
        ),
        "gate_record_sha256": gate_record_sha,
        "gate_row_key": gate_row["row_key"],
        "gate_status": gate_row["gate_status"],
        "gold_type": gold_row["gold_type"],
        "mathematically_equivalent": verification["mathematically_equivalent"],
        "metric": METRIC_NAME,
        "native_cap_hit": bool(native_row["cap_hit"]),
        "native_correct": bool(native_row["is_correct"]),
        "problem_id": gate_row["problem_id"],
        "reason_code": decision["reason_code"] if decision is not None else None,
        "route_audit": verification["route_audit"],
        "route_used": verification["route_used"],
        "scorer_version": SCORER_VERSION,
        "shard_index": gate_row["shard_index"],
        "structure_matches": verification["structure_matches"],
        "symbolic_parse_succeeded": verification["symbolic_parse_succeeded"],
        "verification_status": verification["verification_status"],
        "variant": gate_row["variant"],
    }


def repair_score_journal(
    path: Path, gate_rows: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    gates = {str(row["row_key"]): row for row in gate_rows}
    candidates: dict[str, dict[str, Any]] = {}
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_number, line in enumerate(lines, start=1):
            if not line.endswith("\n"):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(lines):
                    break
                raise
            gate_key = str(row.get("gate_row_key"))
            gate = gates.get(gate_key)
            expected_gate_sha = (
                sha256_text(native.controller.canonical_json(gate)) if gate is not None else None
            )
            valid = (
                gate is not None
                and row.get("metric") == METRIC_NAME
                and row.get("scorer_version") == SCORER_VERSION
                and row.get("gate_record_sha256") == expected_gate_sha
                and row.get("problem_id") == gate["problem_id"]
                and row.get("variant") == gate["variant"]
                and row.get("shard_index") == gate["shard_index"]
                and (type(row.get("correct")) is bool or row.get("correct") is None)
            )
            if not valid:
                if gate is not None and row.get("gate_record_sha256") != expected_gate_sha:
                    continue
                if line_number == len(lines):
                    break
                raise RuntimeError(f"invalid score journal row at {path}:{line_number}")
            candidates[gate_key] = row
    ordered = [candidates[str(gate["row_key"])] for gate in gate_rows if str(gate["row_key"]) in candidates]
    if path.exists() or ordered:
        native.controller.atomic_text(path, native.controller.jsonl_text(ordered))
    return {str(row["gate_row_key"]): row for row in ordered}


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        raise RuntimeError("cannot summarize zero score rows")
    resolved = sum(type(row["correct"]) is bool for row in rows)
    unresolved = total - resolved
    correct = sum(row["correct"] is True for row in rows)
    definitive_known = sum(row["definitive"] is not None for row in rows)
    definitive = sum(row["definitive"] is True for row in rows)
    parsed = sum(row["symbolic_parse_succeeded"] is True for row in rows)
    format_known = sum(row["format_compliant"] is not None for row in rows)
    format_compliant = sum(row["format_compliant"] is True for row in rows)
    cap_hits = sum(bool(row["native_cap_hit"]) for row in rows)
    unresolved_cap_hits = sum(
        bool(row["native_cap_hit"]) and row["correct"] is None for row in rows
    )
    known_cap_hits_correct = sum(
        bool(row["native_cap_hit"]) and row["correct"] is True for row in rows
    )
    unresolved_non_cap = sum(
        not bool(row["native_cap_hit"]) and row["correct"] is None for row in rows
    )
    known_correct_non_cap = correct - known_cap_hits_correct
    all_cap_wrong_lower = known_correct_non_cap
    all_cap_wrong_upper = known_correct_non_cap + unresolved_non_cap
    all_cap_right_lower = known_correct_non_cap + cap_hits
    all_cap_right_upper = all_cap_right_lower + unresolved_non_cap
    transitions = Counter(
        f"native_{int(bool(row['native_correct']))}_strict_{int(bool(row['correct']))}"
        for row in rows if type(row["correct"]) is bool
    )
    return {
        "accuracy": correct / total if not unresolved else None,
        "accuracy_lower_bound": correct / total,
        "accuracy_upper_bound": (correct + unresolved) / total,
        "cap_hit_count": cap_hits,
        "cap_hit_rate": cap_hits / total,
        "cap_hit_percent": 100 * cap_hits / total,
        "cap_hits_correct": known_cap_hits_correct if not unresolved_cap_hits else None,
        "cap_hits_correct_lower_bound": known_cap_hits_correct,
        "cap_hits_correct_upper_bound": known_cap_hits_correct + unresolved_cap_hits,
        "cap_hits_unresolved": unresolved_cap_hits,
        "correct": correct,
        "correct_if_all_cap_hits_right": (
            all_cap_right_lower if not unresolved_non_cap else None
        ),
        "correct_if_all_cap_hits_right_lower_bound": all_cap_right_lower,
        "correct_if_all_cap_hits_right_rate": (
            all_cap_right_lower / total if not unresolved_non_cap else None
        ),
        "correct_if_all_cap_hits_right_rate_lower_bound": all_cap_right_lower / total,
        "correct_if_all_cap_hits_right_rate_upper_bound": all_cap_right_upper / total,
        "correct_if_all_cap_hits_right_upper_bound": all_cap_right_upper,
        "correct_if_all_cap_hits_wrong": (
            all_cap_wrong_lower if not unresolved_non_cap else None
        ),
        "correct_if_all_cap_hits_wrong_lower_bound": all_cap_wrong_lower,
        "correct_if_all_cap_hits_wrong_rate": (
            all_cap_wrong_lower / total if not unresolved_non_cap else None
        ),
        "correct_if_all_cap_hits_wrong_rate_lower_bound": all_cap_wrong_lower / total,
        "correct_if_all_cap_hits_wrong_rate_upper_bound": all_cap_wrong_upper / total,
        "correct_if_all_cap_hits_wrong_upper_bound": all_cap_wrong_upper,
        "correct_unboxed": sum(
            row["correct"] is True and row["format_compliant"] is False for row in rows
        ),
        "definitive": definitive,
        "definitive_rate": definitive / total if definitive_known == total else None,
        "format_compliant": format_compliant,
        "format_compliance_rate": format_compliant / total if format_known == total else None,
        "native_to_strict_transitions": dict(sorted(transitions.items())),
        "provisional_accuracy_on_resolved": correct / resolved if resolved else None,
        "resolved": resolved,
        "scoring_coverage": resolved / total,
        "symbolic_parse_rate": parsed / definitive if definitive else 0.0,
        "symbolic_parse_succeeded": parsed,
        "total": total,
        "unresolved": unresolved,
        "unresolved_non_cap": unresolved_non_cap,
    }


def score(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    value = inventory(repo_root)
    paths = roots(repo_root)
    gate_rows = all_gate_rows(repo_root)
    gold_rows = native.controller.read_jsonl(paths["gold"])
    gold_by_id = {str(row["problem_id"]): row for row in gold_rows}
    if len(gold_rows) != 500 or len(gold_by_id) != 500:
        raise RuntimeError("gold structure manifest must contain 500 unique rows")
    native_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for item in value["checkpoints"]:
        variant = str(item["variant"])
        output = native.controller.absolute_item_paths(repo_root, item)["output"]
        native_rows = native.controller.read_jsonl(output / "merged/predictions.jsonl")
        native_by_variant[variant] = {str(row["problem_id"]): row for row in native_rows}
        if len(native_rows) != 500 or len(native_by_variant[variant]) != 500:
            raise RuntimeError(f"native predictions are incomplete: {variant}")
    completed = repair_score_journal(paths["scores"], gate_rows)
    ordered_scores: list[dict[str, Any]] = []
    for gate_row in gate_rows:
        gate_key = str(gate_row["row_key"])
        existing = completed.get(gate_key)
        if existing is None:
            problem_id = str(gate_row["problem_id"])
            variant = str(gate_row["variant"])
            existing = score_row(
                gate_row, gold_by_id[problem_id], native_by_variant[variant][problem_id]
            )
            append_fsynced(paths["scores"], existing)
        ordered_scores.append(existing)
    native.controller.atomic_text(paths["scores"], native.controller.jsonl_text(ordered_scores))
    per_variant: list[dict[str, Any]] = []
    for item in value["checkpoints"]:
        variant = str(item["variant"])
        rows = [row for row in ordered_scores if row["variant"] == variant]
        variant_metrics = metrics(rows)
        summary = {
            "artifact_schema_version": GATE_ARTIFACT_VERSION,
            "benchmark": "HuggingFaceH4/MATH-500",
            "dataset_revision": value["dataset"]["revision"],
            "metric": METRIC_NAME,
            "scorer_version": SCORER_VERSION,
            **variant_metrics,
            "per_problem": rows,
            "variant": variant,
        }
        destination = result_path(repo_root, item)
        native.controller.atomic_text(
            destination / "RESULTS.json", native.controller.canonical_json(summary)
        )
        native.controller.atomic_text(
            destination / "RESULTS.sha256",
            native.controller.sha256_file(destination / "RESULTS.json") + "\n",
        )
        per_variant.append(
            {
                "result": str(destination),
                "variant": variant,
                **variant_metrics,
            }
        )
    suite_metrics = metrics(ordered_scores)
    suite = {
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "completed_at": native.controller.now_iso(),
        "metric": METRIC_NAME,
        "scorer_version": SCORER_VERSION,
        **suite_metrics,
        "results": per_variant,
        "status": "complete" if suite_metrics["unresolved"] == 0 else "incomplete",
    }
    native.controller.atomic_text(paths["suite_result"], native.controller.canonical_json(suite))
    print(
        f"LLAMA_STRICT_SCORE_PUBLISHED status={suite['status']} "
        f"correct={suite['correct']}/{suite['total']} unresolved={suite['unresolved']} "
        f"cap_hits={suite['cap_hit_count']} cap_hits_correct={suite['cap_hits_correct']}",
        flush=True,
    )


def check_runtime(_args: argparse.Namespace) -> None:
    expected = policy()["runtime"]
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Llama gate runtime mismatch: {actual} != {expected}")
    print("LLAMA_GATE_RUNTIME_VERSIONS_VALID " + " ".join(f"{k}={v}" for k, v in actual.items()))


def check_model(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    ready = validate_model(repo_root)
    print(
        f"LLAMA_GATE_MODEL_VALID revision={ready['revision']} shards={len(ready['weight_shards'])}",
        flush=True,
    )


def audit(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    value = inventory(repo_root)
    paths = roots(repo_root)
    metadata = native.controller.load_json(paths["gate_metadata"])
    if metadata.get("rows") != EXPECTED_ROWS or metadata.get("shards") != 48:
        raise RuntimeError("gate metadata is incomplete")
    for variant in variants(value):
        for shard_index in range(EXPECTED_SHARDS):
            input_rows = native.controller.read_jsonl(input_path(repo_root, variant, shard_index))
            if len(input_rows) != ROWS_PER_SHARD:
                raise RuntimeError(f"incomplete gate input: {variant}/{shard_index}")
            if any(set(row) != ALLOWED_GATE_INPUT_FIELDS for row in input_rows):
                raise RuntimeError(f"gold-bearing or unknown gate field: {variant}/{shard_index}")
    gate_rows = all_gate_rows(repo_root)
    score_rows = native.controller.read_jsonl(paths["scores"])
    if len(score_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected 6000 scores, found {len(score_rows)}")
    gate_keys = {str(row["row_key"]) for row in gate_rows}
    score_keys = {str(row.get("gate_row_key")) for row in score_rows}
    if len(gate_keys) != EXPECTED_ROWS or gate_keys != score_keys:
        raise RuntimeError("gate and score row identities differ")
    suite = native.controller.load_json(paths["suite_result"])
    if suite.get("status") not in {"complete", "incomplete"} or suite.get("total") != EXPECTED_ROWS:
        raise RuntimeError("suite result has an invalid publication state")
    if suite.get("status") == "complete" and suite.get("unresolved") != 0:
        raise RuntimeError("complete suite still contains unresolved rows")
    if suite.get("status") == "incomplete" and not suite.get("unresolved"):
        raise RuntimeError("incomplete suite does not report unresolved rows")
    for item in value["checkpoints"]:
        result = native.controller.load_json(result_path(repo_root, item) / "RESULTS.json")
        if result.get("total") != 500 or len(result.get("per_problem", [])) != 500:
            raise RuntimeError(f"incomplete strict result: {item['variant']}")
        for field in (
            "cap_hit_percent",
            "cap_hits_correct",
            "correct_if_all_cap_hits_wrong",
            "correct_if_all_cap_hits_right",
        ):
            if field not in result:
                raise RuntimeError(f"strict result lacks {field}: {item['variant']}")
    final = {
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "audited_at": native.controller.now_iso(),
        "gate_rows": len(gate_rows),
        "metric": METRIC_NAME,
        "score_rows": len(score_rows),
        "status": suite["status"],
        "suite_result": str(paths["suite_result"]),
        "variants": EXPECTED_VARIANTS,
    }
    audit_name = "FINAL_AUDIT.json" if suite["status"] == "complete" else "PROVISIONAL_AUDIT.json"
    native.controller.atomic_text(
        paths["scoring"] / audit_name, native.controller.canonical_json(final)
    )
    print(
        f"LLAMA_STRICT_AUDIT_PUBLISHED status={suite['status']} "
        f"rows={EXPECTED_ROWS} unresolved={suite['unresolved']} variants=12",
        flush=True,
    )


def record_submission(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    value = inventory(repo_root)
    assignments: dict[str, str] = {}
    for assignment in args.gate_array_job:
        variant, separator, job_id = assignment.partition("=")
        if not separator or variant in assignments or variant not in variants(value):
            raise RuntimeError(f"invalid gate array assignment: {assignment}")
        if not job_id.isdigit():
            raise RuntimeError(f"invalid Slurm job ID: {job_id}")
        assignments[variant] = job_id
    if set(assignments) != set(variants(value)):
        raise RuntimeError("submission must record one four-shard gate array per variant")
    for job_id in (args.prepare_job, args.canary_job, args.score_job, args.audit_job):
        if not str(job_id).isdigit():
            raise RuntimeError(f"invalid Slurm job ID: {job_id}")
    destination = roots(repo_root)["gate"] / "submission.json"
    if destination.exists():
        raise RuntimeError(f"refusing to replace submission journal: {destination}")
    submission = {
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "audit_job": str(args.audit_job),
        "canary_job": str(args.canary_job),
        "gate_array_jobs": assignments,
        "gate_shards": 48,
        "model": policy()["model"],
        "policy_sha256": gate_policy_sha256(),
        "prepare_job": str(args.prepare_job),
        "recorded_at": native.controller.now_iso(),
        "score_job": str(args.score_job),
    }
    native.controller.atomic_text(destination, native.controller.canonical_json(submission))
    print(f"LLAMA_GATE_SUBMISSION_RECORDED path={destination}", flush=True)


def record_preflight_recovery(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    paths = roots(repo_root)
    native_submission = native.controller.load_json(paths["native_control"] / "submission.json")
    if str(native_submission.get("preflight_job")) != str(args.failed_job):
        raise RuntimeError("failed preflight does not match the submitted native DAG")
    if not str(args.replacement_job).isdigit():
        raise RuntimeError("replacement preflight job ID is invalid")
    destination = paths["native_control"] / "PREFLIGHT_RECOVERY.json"
    if destination.exists():
        raise RuntimeError(f"refusing to replace preflight recovery journal: {destination}")
    value = {
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "failed_job": str(args.failed_job),
        "failure": "fresh runtime lacked the evaluator's pyparsing dependency",
        "generation_array_jobs_rewired": native_submission["array_jobs"],
        "policy_sha256_after_repair": gate_policy_sha256(),
        "recorded_at": native.controller.now_iso(),
        "repair": "pinned and installed pyparsing==3.3.2; no inference task had started",
        "replacement_job": str(args.replacement_job),
    }
    native.controller.atomic_text(destination, native.controller.canonical_json(value))
    print(f"LLAMA_PREFLIGHT_RECOVERY_RECORDED path={destination}", flush=True)


def show_path(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    available = roots(repo_root)
    if args.field == "input":
        if args.variant is None or args.shard_index is None:
            raise RuntimeError("input path requires --variant and --shard-index")
        result = input_path(repo_root, args.variant, args.shard_index)
    elif args.field == "decision":
        if args.variant is None or args.shard_index is None:
            raise RuntimeError("decision path requires --variant and --shard-index")
        result = decision_path(repo_root, args.variant, args.shard_index)
    else:
        result = available[args.field]
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-inputs")
    prepare_parser.set_defaults(func=prepare_inputs)

    gate_parser = subparsers.add_parser("run-gate")
    gate_parser.add_argument("--variant", required=True)
    gate_parser.add_argument("--shard-index", required=True, type=int)
    gate_parser.add_argument("--overlay-id")
    gate_parser.set_defaults(func=run_gate)

    canary_parser = subparsers.add_parser("run-canary")
    canary_parser.set_defaults(func=run_canary)

    score_parser = subparsers.add_parser("score")
    score_parser.set_defaults(func=score)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.set_defaults(func=audit)

    runtime_parser = subparsers.add_parser("check-runtime")
    runtime_parser.set_defaults(func=check_runtime)

    model_parser = subparsers.add_parser("check-model")
    model_parser.set_defaults(func=check_model)

    path_parser = subparsers.add_parser("path")
    path_parser.add_argument(
        "--field",
        choices=(
            "decision",
            "gate",
            "gate_inputs",
            "gate_metadata",
            "gate_outputs",
            "canary_complete",
            "canary_inputs",
            "canary_outputs",
            "gold",
            "input",
            "model",
            "native_control",
            "python",
            "runtime",
            "scores",
            "scoring",
            "suite_result",
        ),
        required=True,
    )
    path_parser.add_argument("--variant")
    path_parser.add_argument("--shard-index", type=int)
    path_parser.set_defaults(func=show_path)

    submission_parser = subparsers.add_parser("record-submission")
    submission_parser.add_argument("--prepare-job", required=True)
    submission_parser.add_argument("--canary-job", required=True)
    submission_parser.add_argument("--gate-array-job", action="append", default=[])
    submission_parser.add_argument("--score-job", required=True)
    submission_parser.add_argument("--audit-job", required=True)
    submission_parser.set_defaults(func=record_submission)

    recovery_parser = subparsers.add_parser("record-preflight-recovery")
    recovery_parser.add_argument("--failed-job", required=True)
    recovery_parser.add_argument("--replacement-job", required=True)
    recovery_parser.set_defaults(func=record_preflight_recovery)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
