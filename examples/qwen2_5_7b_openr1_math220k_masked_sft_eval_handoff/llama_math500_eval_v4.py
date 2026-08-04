#!/usr/bin/env python3
"""Source-mapped Llama extraction and strict MATH-500 rescoring.

The current-corpus preparation command imports only the unchanged v2
adjudication and evidence-region decision.  Every definitive response receives
a v4 extraction. Candidate answers are mapped back to exact source text;
free-form fallback answers must be present modulo presentation formatting.
GPU commands have no path to the gold manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import re
import signal
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any, Sequence

from examples.qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff import (
    llama_math500_eval as v2,
    llama_math500_eval_v3 as v3,
)


native = v2.native
HANDOFF = Path(__file__).resolve().parent
POLICY_PATH = HANDOFF / "llama_gate_v4_policy.json"
GATE_ARTIFACT_VERSION = 4
GATE_IMPLEMENTATION_VERSION = "llama-source-mapped-v4.0.0"
EXTRACTION_CATALOG_VERSION = "source-mapped-catalog-v4.0.0"
FORMAT_MATCH_VERSION = "math-answer-presentation-v1.0.0"
SCORER_VERSION = "strict-router-v4.0.0"
METRIC_NAME = "math500_llama33_source_mapped_strict_v4"
EXPECTED_VARIANTS = v2.EXPECTED_VARIANTS
EXPECTED_SHARDS = v2.EXPECTED_SHARDS
ROWS_PER_SHARD = v2.ROWS_PER_SHARD
EXPECTED_ROWS = v2.EXPECTED_ROWS
SELECTION_MODES = {"boxed_exact", "heuristic_exact", "verbatim_region"}
EXTRACTION_FIELDS = {"selection_mode", "final_answer_raw"}
V4_INPUT_FIELDS = v2.ALLOWED_GATE_INPUT_FIELDS | {
    "adjudication",
    "adjudication_attempts",
    "adjudication_source",
    "adjudication_unresolved",
}


def policy() -> dict[str, Any]:
    return native.controller.load_json(POLICY_PATH)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_text(native.controller.canonical_json(value))


def adjudication_policy_manifest() -> dict[str, Any]:
    value = policy()
    prompt = str(value["gate"]["adjudication_prompt"])
    return {
        "adjudication_fields": sorted(v2.ADJUDICATION_FIELDS),
        "adjudication_schema_version": v2.GATE_IMPLEMENTATION_VERSION,
        "engine": value["engine"],
        "model": value["model"],
        "prompt_sha256": native.controller.sha256_file(HANDOFF / prompt),
        "response_region_catalog_version": v2.SPAN_CATALOG_VERSION,
        "response_format": value["gate"]["response_format"],
        "temperature": value["gate"]["temperature"],
    }


def adjudication_policy_sha256() -> str:
    return canonical_sha(adjudication_policy_manifest())


def extraction_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(EXTRACTION_FIELDS),
        "properties": {
            "selection_mode": {"type": "string", "enum": sorted(SELECTION_MODES)},
            "final_answer_raw": {"type": "string", "minLength": 1},
        },
    }


def gate_policy_manifest() -> dict[str, Any]:
    value = policy()
    prompt = str(value["gate"]["extraction_prompt"])
    runtime_names = {"llguidance", "torch", "transformers", "vllm", "xgrammar"}
    return {
        "adjudication_policy": adjudication_policy_manifest(),
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "extraction_catalog_version": EXTRACTION_CATALOG_VERSION,
        "format_match_version": FORMAT_MATCH_VERSION,
        "extraction_schema": extraction_schema(),
        "gate_implementation_version": GATE_IMPLEMENTATION_VERSION,
        "policy": {
            "engine": value["engine"],
            "gate": value["gate"],
            "model": value["model"],
            "runtime": {
                name: version
                for name, version in value["runtime"].items()
                if name in runtime_names
            },
        },
        "prompt_sha256": native.controller.sha256_file(HANDOFF / prompt),
        "source_span_catalog_version": v2.SPAN_CATALOG_VERSION,
    }


def gate_policy_sha256() -> str:
    return canonical_sha(gate_policy_manifest())


def inventory(repo_root: Path) -> dict[str, Any]:
    return v2.inventory(repo_root)


def variants(value: dict[str, Any]) -> list[str]:
    return v2.variants(value)


def roots(repo_root: Path) -> dict[str, Path]:
    value = inventory(repo_root)
    evaluation = native.controller.eval_root(repo_root, value)
    gate = evaluation / "_llama_gate_v4_no_gold"
    scoring = evaluation / "_llama_strict_scoring_v4"
    v2_paths = v2.roots(repo_root)
    return {
        "audit": scoring / "PROVISIONAL_AUDIT.json",
        "canary_complete": gate / "canary.complete.json",
        "canary_inputs": gate / "canary_inputs.jsonl",
        "canary_outputs": gate / "canary_outputs.jsonl",
        "gate": gate,
        "gate_inputs": gate / "gate_inputs",
        "gate_metadata": gate / "metadata.json",
        "gate_outputs": gate / "gate_outputs",
        "gold": v2_paths["gold"],
        "model": v2_paths["model"],
        "native_control": v2_paths["native_control"],
        "python": v2_paths["python"],
        "review": scoring / "UNRESOLVED_REVIEW.jsonl",
        "runtime": v2_paths["runtime"],
        "scores": scoring / "scores.jsonl",
        "scoring": scoring,
        "suite_result": scoring / "RESULTS.json",
        "table": scoring / "PROVISIONAL_TABLE.md",
    }


def input_path(repo_root: Path, variant: str, shard_index: int) -> Path:
    return roots(repo_root)["gate_inputs"] / variant / f"shard_{shard_index:02d}.jsonl"


def decision_path(repo_root: Path, variant: str, shard_index: int) -> Path:
    return roots(repo_root)["gate_outputs"] / variant / f"shard_{shard_index:02d}.jsonl"


def decision_complete_path(repo_root: Path, variant: str, shard_index: int) -> Path:
    return roots(repo_root)["gate_outputs"] / variant / f"shard_{shard_index:02d}.complete.json"


def result_path(repo_root: Path, item: dict[str, Any]) -> Path:
    native_result = native.controller.absolute_item_paths(repo_root, item)["result"]
    return native_result / "rescored" / METRIC_NAME


def source_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["variant"]), str(row["problem_id"]), str(row["response_sha256"])


def source_key(row: dict[str, Any]) -> str:
    return "\0".join(source_identity(row))


def row_key(row: dict[str, Any]) -> str:
    return sha256_text("\0".join((*source_identity(row), gate_policy_sha256())))


def append_fsynced(path: Path, row: dict[str, Any]) -> None:
    v2.append_fsynced(path, row)


def prepare_inputs(args: argparse.Namespace) -> None:
    """One-off import of the unchanged v2 adjudication stage."""
    repo_root = native.controller.repo_root_from_args(args)
    value = inventory(repo_root)
    expected = policy()["one_off_reuse"]
    if v2.gate_policy_sha256() != expected["source_policy_sha256"]:
        raise RuntimeError("the pinned v2 source policy is not the policy being imported")
    old_rows = v2.all_gate_rows(repo_root)
    old_by_source = {source_key(row): row for row in old_rows}
    if len(old_by_source) != EXPECTED_ROWS:
        raise RuntimeError("v2 source gate is not exactly 6000 unique rows")

    counts = {"adjudications": 0, "definitive": 0, "missing": 0, "nondefinitive": 0}
    prepared: list[dict[str, Any]] = []
    canary_ranked: list[tuple[int, str, dict[str, Any]]] = []
    for variant in variants(value):
        for shard_index in range(EXPECTED_SHARDS):
            source_rows, source_policy = v2.validated_gate_inputs(
                repo_root, variant, shard_index
            )
            if source_policy != expected["source_policy_sha256"]:
                raise RuntimeError("v2 input shard policy differs from the pinned import")
            shard_rows: list[dict[str, Any]] = []
            for source in source_rows:
                old = old_by_source[source_key(source)]
                adjudication = old.get("adjudication")
                unresolved = None
                if adjudication is None:
                    counts["missing"] += 1
                    unresolved = old.get("unresolved") or {
                        "stage": "adjudication",
                        "last_error": "v2 adjudication unavailable",
                    }
                else:
                    regions = v2.response_regions(str(source["response"]))
                    adjudication = v2.validate_adjudication(adjudication, regions)
                    counts["adjudications"] += 1
                    if adjudication["definitive"]:
                        counts["definitive"] += 1
                    else:
                        counts["nondefinitive"] += 1
                imported = {
                    **source,
                    "adjudication": adjudication,
                    "adjudication_attempts": int(old.get("attempts", {}).get("adjudication", 0)),
                    "adjudication_source": {
                        "artifact": "strict-v2-one-off-reuse",
                        "gate_record_sha256": canonical_sha(old),
                        "policy_sha256": source_policy,
                        "row_key": old["row_key"],
                    },
                    "adjudication_unresolved": unresolved,
                }
                if set(imported) != V4_INPUT_FIELDS:
                    raise RuntimeError("v4 imported input fields drifted")
                shard_rows.append(imported)
                prepared.append(imported)
                if adjudication is not None and adjudication["definitive"]:
                    old_mode = (old.get("extraction") or {}).get("extraction_mode")
                    priority = 0 if old["gate_status"] == "unresolved" else 1 if old_mode == "boundary" else 2
                    canary_ranked.append((priority, source_key(source), imported))
            if len(shard_rows) != ROWS_PER_SHARD:
                raise RuntimeError("v4 import did not preserve 125-row shards")
            native.controller.atomic_text(
                input_path(repo_root, variant, shard_index),
                native.controller.jsonl_text(shard_rows),
            )

    observed = {
        "expected_adjudications": counts["adjudications"],
        "expected_definitive": counts["definitive"],
        "expected_missing_adjudications": counts["missing"],
        "expected_nondefinitive": counts["nondefinitive"],
    }
    for name, count in observed.items():
        if count != int(expected[name]):
            raise RuntimeError(f"one-off import count mismatch for {name}: {count}")
    if len(prepared) != EXPECTED_ROWS:
        raise RuntimeError("v4 import did not prepare exactly 6000 rows")

    canary = [row for _, _, row in sorted(canary_ranked, key=lambda item: (item[0], item[1]))[:64]]
    paths = roots(repo_root)
    native.controller.atomic_text(paths["canary_inputs"], native.controller.jsonl_text(canary))
    metadata = {
        "adjudication_policy_sha256": adjudication_policy_sha256(),
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "canary_rows": len(canary),
        "created_at": native.controller.now_iso(),
        "gate_input_fields": sorted(V4_INPUT_FIELDS),
        "gold_location": "intentionally omitted from GPU artifacts",
        "one_off_reuse": {**observed, "source_policy_sha256": expected["source_policy_sha256"]},
        "policy_sha256": gate_policy_sha256(),
        "rows": EXPECTED_ROWS,
        "shards": EXPECTED_VARIANTS * EXPECTED_SHARDS,
    }
    native.controller.atomic_text(paths["gate_metadata"], native.controller.canonical_json(metadata))
    print(
        "LLAMA_V4_INPUTS_PREPARED "
        f"rows={EXPECTED_ROWS} adjudications={counts['adjudications']} "
        f"definitive={counts['definitive']} missing={counts['missing']} canary={len(canary)}",
        flush=True,
    )


def validated_inputs(
    repo_root: Path, variant: str, shard_index: int
) -> tuple[list[dict[str, Any]], str]:
    value = inventory(repo_root)
    if variant not in variants(value) or shard_index not in range(EXPECTED_SHARDS):
        raise RuntimeError(f"unknown v4 shard: {variant}/{shard_index}")
    metadata = native.controller.load_json(roots(repo_root)["gate_metadata"])
    policy_sha = gate_policy_sha256()
    if metadata.get("policy_sha256") != policy_sha:
        raise RuntimeError("prepared v4 policy differs from the running code")
    if metadata.get("gate_input_fields") != sorted(V4_INPUT_FIELDS):
        raise RuntimeError("v4 input allow-list mismatch")
    rows = native.controller.read_jsonl(input_path(repo_root, variant, shard_index))
    if len(rows) != ROWS_PER_SHARD:
        raise RuntimeError("v4 shard is not 125 rows")
    for row in rows:
        if set(row) != V4_INPUT_FIELDS:
            raise RuntimeError("v4 shard contains forbidden or missing fields")
        if row["variant"] != variant or int(row["shard_index"]) != shard_index:
            raise RuntimeError("v4 shard identity mismatch")
        if sha256_text(str(row["response"])) != row["response_sha256"]:
            raise RuntimeError("v4 response hash mismatch")
    return rows, policy_sha


def eligible_candidates(
    response: str, region: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = v2.answer_candidates(response, region)
    boxes = v2.balanced_box_spans(response)
    region_start, region_end = int(region["start"]), int(region["end"])
    # Do not inherit v2's 96-candidate display cap for boxes.  A looping trace
    # can contain more than 96 complete boxes and the terminal one must remain
    # selectable even when earlier boxes are all distinct.
    boxed = [
        {
            "boxed": True,
            "end": int(box["content_end"]),
            "origin": "boxed_content",
            "start": int(box["content_start"]),
            "text": response[int(box["content_start"]) : int(box["content_end"])],
        }
        for box in boxes
        if region_start <= int(box["content_start"]) < int(box["content_end"]) <= region_end
    ]
    heuristic: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate["boxed"] or candidate["text"] in seen:
            continue
        start, end = int(candidate["start"]), int(candidate["end"])
        overlaps_box = any(
            start < int(box["wrapper_end"]) and end > int(box["wrapper_start"])
            for box in boxes
        )
        if overlaps_box or r"\boxed" in str(candidate["text"]):
            continue
        seen.add(str(candidate["text"]))
        heuristic.append(candidate)
    return boxed, heuristic


def render_extraction_prompt(
    template: str,
    problem: str,
    adjudication: dict[str, Any],
    region: dict[str, Any],
    boxed: Sequence[dict[str, Any]],
    heuristic: Sequence[dict[str, Any]],
) -> str:
    def menu(values: Sequence[dict[str, Any]]) -> str:
        if not values:
            return "(none)"
        lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for value in values:
            identity = str(value["origin"]), str(value["text"])
            if identity in seen:
                continue
            seen.add(identity)
            lines.append(
                f"- origin={value['origin']}: {json.dumps(value['text'], ensure_ascii=False)}"
            )
        return "\n".join(lines)

    replacements = {
        "{{PROBLEM}}": problem,
        "{{ADJUDICATION}}": json.dumps(adjudication, ensure_ascii=False, sort_keys=True),
        "{{EVIDENCE}}": json.dumps(region["text"], ensure_ascii=False),
        "{{BOXED_CANDIDATES}}": menu(boxed),
        "{{HEURISTIC_CANDIDATES}}": menu(heuristic),
    }
    for placeholder, replacement in replacements.items():
        if template.count(placeholder) != 1:
            raise RuntimeError(f"v4 prompt placeholder changed: {placeholder}")
        template = template.replace(placeholder, replacement)
    return template


_COMMANDS_WITH_OPTIONAL_SLASH = (
    "begin|cdot|circ|cos|dfrac|displaystyle|end|frac|in|infty|left|ln|log|"
    "mathrm|mathbf|mathit|operatorname|pi|pm|qquad|quad|right|sin|sqrt|tan|"
    "text|tfrac|theta|times"
)


def _strip_outer_braced_command(value: str, command: str) -> str:
    match = re.match(rf"^\\+{command}\s*\{{", value)
    if match is None:
        return value
    depth = 1
    for index in range(match.end(), len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                if not value[index + 1 :].strip():
                    return value[match.end() : index].strip()
                break
    return value


def strip_presentation_wrappers(value: str) -> str:
    """Remove wrappers observed in prior constrained extraction attempts."""
    result = html.unescape(value)
    if re.search(r"%[0-9A-Fa-f]{2}", result):
        result = urllib.parse.unquote(result)
    result = unicodedata.normalize("NFKC", result).replace("¥", "\\").strip()
    changed = True
    while changed:
        before = result
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        result = result.strip("`").strip()
        if result.startswith("<boxed>") and result.endswith("</boxed>"):
            result = result[len("<boxed>") : -len("</boxed>")].strip()
        for opening, closing in ((r"\(", r"\)"), (r"\[", r"\]"), ("$", "$")):
            if result.startswith(opening) and result.endswith(closing):
                result = result[len(opening) : -len(closing)].strip()
        result = _strip_outer_braced_command(result, "boxed")
        changed = result != before
    return result


def _replace_latex_groups(value: str) -> str:
    """Canonicalize presentation commands without evaluating mathematics."""
    result = value
    flat = r"([^{}]*)"
    previous = None
    while result != previous:
        previous = result
        result = re.sub(
            rf"(?:text|mathrm|mathbf|mathit|operatorname)\{{{flat}\}}",
            lambda match: match.group(1),
            result,
        )
        result = re.sub(
            rf"sqrt\{{{flat}\}}",
            lambda match: f"sqrt({match.group(1)})",
            result,
        )
        result = re.sub(
            rf"frac\{{{flat}\}}\{{{flat}\}}",
            lambda match: f"div({match.group(1)},{match.group(2)})",
            result,
        )
    result = re.sub(r"sqrt([A-Za-z0-9.]+)", r"sqrt(\1)", result)
    result = re.sub(
        r"(?<![A-Za-z0-9}])([+-]?[A-Za-z0-9.]+)/([+-]?[A-Za-z0-9.]+)",
        r"div(\1,\2)",
        result,
    )
    return result


def presentation_key(value: str) -> str:
    """Return a non-solving key that ignores observed answer presentation drift."""
    result = strip_presentation_wrappers(value)
    result = result.replace("π", "pi").replace("∞", "infty").replace("√", "sqrt")
    result = result.replace("−", "-").replace("–", "-").replace("—", "-")
    result = result.replace("×", "*").replace("·", "*").replace("÷", "/")
    result = result.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    result = result.replace("∈", "in")
    result = result.replace("±", "pm").replace("°", "^circ")
    result = re.sub(r"\\+(?=(?:" + _COMMANDS_WITH_OPTIONAL_SLASH + r")\b)", "", result)
    result = re.sub(r"\b(?:dfrac|tfrac)\b", "frac", result)
    result = re.sub(r"\b(?:left|right|displaystyle)\b", "", result)
    result = re.sub(r"\\(?:,|!|:|;)|\b(?:quad|qquad)\b|~", "", result)
    result = result.replace(r"\%", "%").replace(r"\{", "{").replace(r"\}", "}")
    result = result.replace("times", "*").replace("cdot", "*")
    result = _replace_latex_groups(result)
    result = re.sub(r"\^\{([+-]?[A-Za-z0-9.]+)\}", r"^\1", result)
    result = re.sub(r"(?:\^\{\})+\^?circ", "^circ", result)
    result = re.sub(r"\s+|\$|`|\*\*", "", result)
    # Command slashes are gone, so remaining slashes are presentation-level
    # LaTeX row separators. Treat one or more identically.
    result = re.sub(r"\\+", ";", result)
    return result.casefold()


def key_is_present(needle: str, haystack: str) -> bool:
    """Formatting-insensitive containment with token-edge guards."""
    if not needle:
        return False
    cursor = 0
    while True:
        found = haystack.find(needle, cursor)
        if found < 0:
            return False
        end = found + len(needle)
        left_ok = not (
            needle[0].isalnum() and found > 0 and haystack[found - 1].isalnum()
        )
        right_ok = not (
            needle[-1].isalnum() and end < len(haystack) and haystack[end].isalnum()
        )
        if left_ok and right_ok:
            return True
        cursor = found + 1


def matching_candidates(
    generated: str, candidates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    key = presentation_key(generated)
    return [candidate for candidate in candidates if presentation_key(str(candidate["text"])) == key]


def selected_extraction(
    value: Any,
    response: str,
    region: dict[str, Any],
    boxed: Sequence[dict[str, Any]],
    heuristic: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != EXTRACTION_FIELDS:
        raise RuntimeError("v4 extraction does not have the exact schema")
    requested_mode = value.get("selection_mode")
    generated = value.get("final_answer_raw")
    if requested_mode not in SELECTION_MODES:
        raise RuntimeError("unknown v4 selection_mode")
    if not isinstance(generated, str) or not generated.strip():
        raise RuntimeError("v4 final_answer_raw is blank")
    region_start, region_end = int(region["start"]), int(region["end"])
    candidate_pool = boxed if requested_mode == "boxed_exact" else heuristic
    matches = matching_candidates(generated, candidate_pool)
    if requested_mode == "verbatim_region":
        box_matches = matching_candidates(generated, boxed)
        heuristic_matches = matching_candidates(generated, heuristic)
        matches = box_matches or heuristic_matches

    if matches:
        selected = max(matches, key=lambda item: (int(item["start"]), int(item["end"])))
        start, end = int(selected["start"]), int(selected["end"])
        source_raw = response[start:end]
        effective_mode = "boxed_exact" if selected in boxed else "heuristic_exact"
        final_raw = source_raw
        origin = str(selected["origin"])
        content_match = "candidate_presentation_key"
    elif requested_mode == "verbatim_region":
        boxes = v2.balanced_box_spans(response)
        nonbox_parts: list[str] = []
        cursor = region_start
        for box_span in boxes:
            start = max(region_start, int(box_span["wrapper_start"]))
            end = min(region_end, int(box_span["wrapper_end"]))
            if start < end:
                nonbox_parts.append(response[cursor:start])
                cursor = max(cursor, end)
        nonbox_parts.append(response[cursor:region_end])
        if not key_is_present(
            presentation_key(generated), presentation_key("\u241e".join(nonbox_parts))
        ):
            raise RuntimeError(
                "verbatim_region content is not present modulo formatting in the selected region"
            )
        start, end = region_start, region_end
        source_raw = response[start:end]
        effective_mode = "verbatim_region"
        final_raw = generated
        origin = "generated_region_content"
        content_match = "region_presentation_substring"
    else:
        raise RuntimeError(
            f"{requested_mode} content does not match an eligible source candidate"
        )
    if not (region_start <= start < end <= region_end) or response[start:end] != source_raw:
        raise RuntimeError("v4 selected evidence failed source validation")
    return {
        "boxed_content": effective_mode == "boxed_exact",
        "content_match": content_match,
        "end": end,
        "final_answer_raw": final_raw,
        "generated_answer_raw": generated,
        "origin": origin,
        "requested_selection_mode": requested_mode,
        "selection_mode": effective_mode,
        "source_answer_raw": source_raw,
        "start": start,
    }


def terminal_decision(
    response: str,
    adjudication: dict[str, Any],
    regions: Sequence[dict[str, Any]],
    extraction: dict[str, Any] | None,
) -> dict[str, Any]:
    if not adjudication["definitive"]:
        return v2.terminal_decision(response, adjudication, regions, None)
    if extraction is None:
        raise RuntimeError("definitive v4 adjudication lacks extraction")
    start, end = int(extraction["start"]), int(extraction["end"])
    boxed = bool(extraction["boxed_content"])
    reason = str(adjudication["reason_code"])
    if boxed and reason == "single_committed_unboxed_answer":
        reason = "single_committed_answer"
    elif not boxed and reason == "single_committed_answer":
        reason = "single_committed_unboxed_answer"
    decision = {
        "answer_form": adjudication["answer_form"],
        "boxed_final_answer": boxed,
        "definitive": True,
        "final_answer_raw": extraction["final_answer_raw"],
        "format_compliant": boxed,
        "later_conflict_quote": None,
        "reason_code": reason,
        "supporting_quote": v2.exact_excerpt(response, start, end),
    }
    return decision


def validate_gate_row(
    value: Any, source: dict[str, Any], policy_sha: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("v4 gate row is not an object")
    if value.get("policy_sha256") != policy_sha or value.get("row_key") != row_key(source):
        raise RuntimeError("v4 gate row policy or identity mismatch")
    if source_identity(value) != source_identity(source):
        raise RuntimeError("v4 gate row source identity mismatch")
    if value.get("gate_status") not in {"resolved", "unresolved"}:
        raise RuntimeError("v4 gate row status is invalid")
    if value["gate_status"] == "resolved":
        if value.get("decision") is None or value.get("unresolved") is not None:
            raise RuntimeError("resolved v4 row has incompatible terminal fields")
        decision = value["decision"]
        extraction = value.get("extraction")
        if decision.get("definitive"):
            if not isinstance(extraction, dict):
                raise RuntimeError("definitive v4 decision lacks extraction")
            if decision.get("final_answer_raw") != extraction.get("final_answer_raw"):
                raise RuntimeError("v4 decision does not preserve the validated answer")
            if decision.get("supporting_quote") not in str(source["response"]):
                raise RuntimeError("v4 supporting quote is not source evidence")
            if decision.get("format_compliant") != decision.get("boxed_final_answer"):
                raise RuntimeError("v4 boxing fields disagree")
        else:
            v2.validate_decision(decision, str(source["response"]))
    else:
        if value.get("decision") is not None or not isinstance(value.get("unresolved"), dict):
            raise RuntimeError("unresolved v4 row has incompatible terminal fields")
    extraction = value.get("extraction")
    if extraction is not None:
        start, end = extraction.get("start"), extraction.get("end")
        source_raw = extraction.get("source_answer_raw")
        if (
            type(start) is not int
            or type(end) is not int
            or str(source["response"])[start:end] != source_raw
        ):
            raise RuntimeError("v4 extraction evidence is not an exact response span")
    return value


def repair_journal(
    path: Path, source_rows: Sequence[dict[str, Any]], policy_sha: str
) -> dict[str, dict[str, Any]]:
    sources = {source_key(row): row for row in source_rows}
    completed: dict[str, dict[str, Any]] = {}
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_number, line in enumerate(lines, start=1):
            if not line.endswith("\n"):
                break
            try:
                value = json.loads(line)
                key = source_key(value)
                if key not in sources:
                    raise RuntimeError("journal row is not in this input shard")
                completed[key] = validate_gate_row(value, sources[key], policy_sha)
            except (json.JSONDecodeError, KeyError, RuntimeError):
                if line_number == len(lines):
                    break
                raise
    ordered = [completed[source_key(row)] for row in source_rows if source_key(row) in completed]
    if path.exists() or ordered:
        native.controller.atomic_text(path, native.controller.jsonl_text(ordered))
    return {source_key(row): row for row in ordered}


def make_gate_row(
    source: dict[str, Any],
    extraction: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    extraction_attempts: int,
    unresolved: dict[str, Any] | None,
    policy_sha: str,
) -> dict[str, Any]:
    adjudication = source["adjudication"]
    return {
        "adjudication": adjudication,
        "adjudication_source": source["adjudication_source"],
        "attempts": {
            "adjudication": int(source["adjudication_attempts"]),
            "extraction": extraction_attempts,
        },
        "decision": decision,
        "extraction": extraction,
        "gate_status": "unresolved" if unresolved is not None else "resolved",
        "model_revision": policy()["model"]["revision"],
        "policy_sha256": policy_sha,
        "problem_id": source["problem_id"],
        "response_sha256": source["response_sha256"],
        "row_key": row_key(source),
        "shard_index": source["shard_index"],
        "unresolved": unresolved,
        "variant": source["variant"],
    }


def replay_v3_shard(repo_root: Path, variant: str, shard_index: int) -> None:
    """Replay persisted v3 decoded attempts under the v4 CPU-only validator."""
    source_rows, policy_sha = validated_inputs(repo_root, variant, shard_index)
    old_sources, old_policy_sha = v3.validated_inputs(repo_root, variant, shard_index)
    old_rows = v3.repair_journal(
        v3.decision_path(repo_root, variant, shard_index), old_sources, old_policy_sha
    )
    failures_by_source: dict[str, list[dict[str, Any]]] = {}
    failure_path = (
        v3.roots(repo_root)["gate_outputs"]
        / "_validation_failures"
        / variant
        / f"shard_{shard_index:02d}.jsonl"
    )
    if failure_path.is_file():
        for failure in native.controller.read_jsonl(failure_path):
            key = "\0".join(
                (
                    str(failure["variant"]),
                    str(failure["problem_id"]),
                    str(failure["response_sha256"]),
                )
            )
            failures_by_source.setdefault(key, []).append(failure)

    replayed: list[dict[str, Any]] = []
    for source in source_rows:
        key = source_key(source)
        old = old_rows[key]
        adjudication = source["adjudication"]
        response = str(source["response"])
        regions = v2.response_regions(response)
        if adjudication is None:
            replayed.append(
                make_gate_row(
                    source,
                    None,
                    None,
                    0,
                    {
                        "attempts": int(source["adjudication_attempts"]),
                        "last_error": source["adjudication_unresolved"],
                        "stage": "adjudication_reuse",
                    },
                    policy_sha,
                )
            )
            continue
        adjudication = v2.validate_adjudication(adjudication, regions)
        if not adjudication["definitive"]:
            replayed.append(
                make_gate_row(
                    source,
                    None,
                    terminal_decision(response, adjudication, regions, None),
                    0,
                    None,
                    policy_sha,
                )
            )
            continue

        region = {item["id"]: item for item in regions}[
            adjudication["evidence_region_id"]
        ]
        boxed, heuristic = eligible_candidates(response, region)
        attempts: list[tuple[int, str]] = [
            (int(item["attempt"]), str(item["decoded"]))
            for item in failures_by_source.get(key, [])
        ]
        old_extraction = old.get("extraction")
        if old_extraction is not None:
            attempts.append(
                (
                    int(old["attempts"]["extraction"]),
                    json.dumps(
                        {
                            "final_answer_raw": old_extraction["final_answer_raw"],
                            "selection_mode": old_extraction["selection_mode"],
                        },
                        ensure_ascii=False,
                    ),
                )
            )

        extraction = None
        last_error = "v3 did not persist a decoded extraction attempt"
        attempts_used = 0
        for attempt_number, decoded in sorted(attempts, key=lambda item: item[0]):
            attempts_used = attempt_number
            try:
                extraction = selected_extraction(
                    json.loads(decoded), response, region, boxed, heuristic
                )
                break
            except (json.JSONDecodeError, RuntimeError) as error:
                last_error = str(error)
        if extraction is None:
            replayed.append(
                make_gate_row(
                    source,
                    None,
                    None,
                    attempts_used,
                    {
                        "attempts": attempts_used,
                        "last_error": last_error,
                        "source_artifact": "strict-v3-decoded-attempt-replay",
                        "stage": "extraction",
                    },
                    policy_sha,
                )
            )
        else:
            replayed.append(
                make_gate_row(
                    source,
                    extraction,
                    terminal_decision(response, adjudication, regions, extraction),
                    attempts_used,
                    None,
                    policy_sha,
                )
            )

    output = decision_path(repo_root, variant, shard_index)
    validated = [
        validate_gate_row(row, source, policy_sha)
        for row, source in zip(replayed, source_rows, strict=True)
    ]
    native.controller.atomic_text(output, native.controller.jsonl_text(validated))
    unresolved = sum(row["gate_status"] == "unresolved" for row in validated)
    native.controller.atomic_text(
        decision_complete_path(repo_root, variant, shard_index),
        native.controller.canonical_json(
            {
                "one_off_reuse": "strict-v3-decoded-attempt-replay",
                "policy_sha256": policy_sha,
                "rows": ROWS_PER_SHARD,
                "status": "complete",
                "unresolved": unresolved,
            }
        ),
    )


def replay_v3_suite(args: argparse.Namespace) -> None:
    """Create and score the current-corpus v4 artifacts without any GPU call."""
    repo_root = native.controller.repo_root_from_args(args)
    prepare_inputs(args)
    for variant in variants(inventory(repo_root)):
        for shard_index in range(EXPECTED_SHARDS):
            replay_v3_shard(repo_root, variant, shard_index)
    score(args)


def execute_extraction_rows(
    repo_root: Path,
    source_rows: Sequence[dict[str, Any]],
    output: Path,
    failure_output: Path,
    policy_sha: str,
) -> dict[str, dict[str, Any]]:
    completed = repair_journal(output, source_rows, policy_sha)
    pending = [row for row in source_rows if source_key(row) not in completed]
    if not pending:
        return completed

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)

    terminal_rows: list[dict[str, Any]] = []
    definitive: list[dict[str, Any]] = []
    template = (HANDOFF / policy()["gate"]["extraction_prompt"]).read_text(encoding="utf-8")
    for source in pending:
        adjudication = source["adjudication"]
        response = str(source["response"])
        regions = v2.response_regions(response)
        if adjudication is None:
            terminal_rows.append(
                make_gate_row(
                    source,
                    None,
                    None,
                    0,
                    {
                        "attempts": int(source["adjudication_attempts"]),
                        "last_error": source["adjudication_unresolved"],
                        "stage": "adjudication_reuse",
                    },
                    policy_sha,
                )
            )
            continue
        adjudication = v2.validate_adjudication(adjudication, regions)
        if not adjudication["definitive"]:
            decision = terminal_decision(response, adjudication, regions, None)
            terminal_rows.append(
                make_gate_row(source, None, decision, 0, None, policy_sha)
            )
            continue
        region_by_id = {region["id"]: region for region in regions}
        region = region_by_id[adjudication["evidence_region_id"]]
        boxed, heuristic = eligible_candidates(response, region)
        content = render_extraction_prompt(
            template,
            str(source["problem"]),
            adjudication,
            region,
            boxed,
            heuristic,
        )
        definitive.append(
            {
                "adjudication": adjudication,
                "boxed": boxed,
                "content": content,
                "heuristic": heuristic,
                "region": region,
                "regions": regions,
                "source": source,
            }
        )

    for row in terminal_rows:
        source = next(item for item in pending if source_key(item) == source_key(row))
        append_fsynced(output, validate_gate_row(row, source, policy_sha))
        completed[source_key(source)] = row

    if not definitive:
        return repair_journal(output, source_rows, policy_sha)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    paths = roots(repo_root)
    gate_policy = policy()
    ready = v2.validate_model(repo_root)
    if ready.get("revision") != gate_policy["model"]["revision"]:
        raise RuntimeError("local Llama model is not the pinned v4 revision")
    tokenizer = AutoTokenizer.from_pretrained(str(paths["model"]), local_files_only=True)
    max_length = int(gate_policy["engine"]["max_model_len"])
    max_tokens = int(gate_policy["gate"]["extraction_max_tokens"])
    rendered: list[dict[str, Any]] = []
    for item in definitive:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["content"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_count = len(tokenizer(chat, add_special_tokens=False)["input_ids"])
        if token_count + max_tokens > max_length:
            source = item["source"]
            unresolved = make_gate_row(
                source,
                None,
                None,
                0,
                {
                    "attempts": 0,
                    "last_error": f"v4 extraction prompt exceeds context: {token_count}+{max_tokens}",
                    "stage": "extraction",
                },
                policy_sha,
            )
            append_fsynced(output, unresolved)
            completed[source_key(source)] = unresolved
        else:
            item["chat"] = chat
            rendered.append(item)

    if not rendered:
        return repair_journal(output, source_rows, policy_sha)

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

    schema = extraction_schema()

    def decode(chats: Sequence[str], tqdm: bool) -> list[str]:
        params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                seed=0,
                guided_decoding=GuidedDecodingParams(json=schema),
            )
            for _ in chats
        ]
        outputs = llm.generate(list(chats), params, use_tqdm=tqdm)
        if len(outputs) != len(chats):
            raise RuntimeError("vLLM returned the wrong v4 batch size")
        return [output.outputs[0].text for output in outputs]

    retries = int(gate_policy["gate"]["semantic_retries"])

    def validate_with_retries(item: dict[str, Any], decoded: str) -> tuple[Any | None, int, str | None]:
        last_error: str | None = None
        for attempt in range(retries + 1):
            try:
                return (
                    selected_extraction(
                        json.loads(decoded),
                        str(item["source"]["response"]),
                        item["region"],
                        item["boxed"],
                        item["heuristic"],
                    ),
                    attempt + 1,
                    None,
                )
            except (json.JSONDecodeError, RuntimeError) as error:
                last_error = str(error)
                source = item["source"]
                append_fsynced(
                    failure_output,
                    {
                        "attempt": attempt + 1,
                        "decoded": decoded,
                        "error": last_error,
                        "problem_id": source["problem_id"],
                        "response_sha256": source["response_sha256"],
                        "stage": "extraction",
                        "variant": source["variant"],
                    },
                )
                if attempt == retries:
                    return None, attempt + 1, last_error
                feedback = (
                    "The previous JSON failed source-content validation: "
                    f"{last_error}. Return corrected JSON only. Use candidate content when "
                    "available; otherwise preserve the answer content represented in the selected "
                    "evidence. Do not use IDs, solve, simplify, or repair."
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
                retry_count = len(tokenizer(retry_chat, add_special_tokens=False)["input_ids"])
                if retry_count + max_tokens > max_length:
                    return None, attempt + 1, "v4 semantic retry exceeds pinned context"
                decoded = decode([retry_chat], False)[0]
        raise AssertionError("unreachable v4 retry loop")

    chunk_size = int(gate_policy["gate"]["chunk_size"])
    for offset in range(0, len(rendered), chunk_size):
        batch = rendered[offset : offset + chunk_size]
        decoded_batch = decode([item["chat"] for item in batch], True)
        for item, decoded in zip(batch, decoded_batch, strict=True):
            source = item["source"]
            extraction, attempts, error = validate_with_retries(item, decoded)
            if extraction is None:
                result = make_gate_row(
                    source,
                    None,
                    None,
                    attempts,
                    {"attempts": attempts, "last_error": error, "stage": "extraction"},
                    policy_sha,
                )
            else:
                decision = terminal_decision(
                    str(source["response"]),
                    item["adjudication"],
                    item["regions"],
                    extraction,
                )
                result = make_gate_row(
                    source, extraction, decision, attempts, None, policy_sha
                )
            append_fsynced(output, validate_gate_row(result, source, policy_sha))
            completed[source_key(source)] = result
        if stop_requested:
            print(f"LLAMA_V4_REQUEUE_SAFE persisted={len(completed)}", flush=True)
            raise SystemExit(75)
    return repair_journal(output, source_rows, policy_sha)


def run_extraction(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    source_rows, policy_sha = validated_inputs(repo_root, args.variant, args.shard_index)
    v2.isolate_vllm_runtime_env(repo_root, f"v4-{args.variant}", args.shard_index)
    output = decision_path(repo_root, args.variant, args.shard_index)
    repaired = execute_extraction_rows(
        repo_root,
        source_rows,
        output,
        roots(repo_root)["gate_outputs"]
        / "_validation_failures"
        / args.variant
        / f"shard_{args.shard_index:02d}.jsonl",
        policy_sha,
    )
    if len(repaired) != ROWS_PER_SHARD:
        raise RuntimeError("v4 extraction shard did not persist all 125 rows")
    unresolved = sum(row["gate_status"] == "unresolved" for row in repaired.values())
    native.controller.atomic_text(
        decision_complete_path(repo_root, args.variant, args.shard_index),
        native.controller.canonical_json(
            {
                "policy_sha256": policy_sha,
                "rows": ROWS_PER_SHARD,
                "status": "complete",
                "unresolved": unresolved,
            }
        ),
    )
    print(
        f"LLAMA_V4_SHARD_COMPLETE variant={args.variant} shard={args.shard_index} "
        f"rows={ROWS_PER_SHARD} unresolved={unresolved}",
        flush=True,
    )


def run_canary(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    paths = roots(repo_root)
    metadata = native.controller.load_json(paths["gate_metadata"])
    rows = native.controller.read_jsonl(paths["canary_inputs"])
    if len(rows) != int(metadata["canary_rows"]):
        raise RuntimeError("v3 canary input is incomplete")
    v2.isolate_vllm_runtime_env(repo_root, "v4-canary", 0)
    repaired = execute_extraction_rows(
        repo_root,
        rows,
        paths["canary_outputs"],
        paths["gate"] / "canary_validation_failures.jsonl",
        str(metadata["policy_sha256"]),
    )
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
    print(f"LLAMA_V4_CANARY_COMPLETE rows={len(rows)} unresolved={unresolved}", flush=True)


def all_gate_rows(repo_root: Path) -> list[dict[str, Any]]:
    value = inventory(repo_root)
    rows: list[dict[str, Any]] = []
    for variant in variants(value):
        for shard_index in range(EXPECTED_SHARDS):
            complete_path = decision_complete_path(repo_root, variant, shard_index)
            if not complete_path.is_file():
                raise RuntimeError(f"v3 shard has no completion marker: {variant}/{shard_index}")
            source_rows, policy_sha = validated_inputs(repo_root, variant, shard_index)
            completed = repair_journal(
                decision_path(repo_root, variant, shard_index), source_rows, policy_sha
            )
            if len(completed) != ROWS_PER_SHARD:
                raise RuntimeError(f"v3 shard is incomplete: {variant}/{shard_index}")
            rows.extend(completed[source_key(source)] for source in source_rows)
    if len(rows) != EXPECTED_ROWS or len({row["row_key"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError("v3 gate is not exactly 6000 unique rows")
    return rows


def score_row(
    gate_row: dict[str, Any], gold_row: dict[str, Any], native_row: dict[str, Any]
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
        verification = v2.verify_answer_with_routing(
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
    extraction = gate_row.get("extraction") or {}
    return {
        "answer_form": decision["answer_form"] if decision is not None else None,
        "boxed_final_answer": decision["boxed_final_answer"] if decision is not None else None,
        "correct": correct,
        "definitive": decision["definitive"] if decision is not None else None,
        "extraction_mode": extraction.get("selection_mode"),
        "final_answer_raw": decision["final_answer_raw"] if decision is not None else None,
        "format_compliant": decision["format_compliant"] if decision is not None else None,
        "gate_record_sha256": canonical_sha(gate_row),
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
    completed: dict[str, dict[str, Any]] = {}
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_number, line in enumerate(lines, start=1):
            if not line.endswith("\n"):
                break
            try:
                value = json.loads(line)
                key = str(value.get("gate_row_key"))
                gate = gates.get(key)
                valid = (
                    gate is not None
                    and value.get("metric") == METRIC_NAME
                    and value.get("scorer_version") == SCORER_VERSION
                    and value.get("gate_record_sha256") == canonical_sha(gate)
                    and (type(value.get("correct")) is bool or value.get("correct") is None)
                )
                if not valid:
                    raise RuntimeError("invalid v3 score journal row")
                completed[key] = value
            except (json.JSONDecodeError, RuntimeError):
                if line_number == len(lines):
                    break
                raise
    ordered = [completed[str(gate["row_key"])] for gate in gate_rows if str(gate["row_key"]) in completed]
    if path.exists() or ordered:
        native.controller.atomic_text(path, native.controller.jsonl_text(ordered))
    return {str(row["gate_row_key"]): row for row in ordered}


def format_count_range(lower: int, upper: int, total: int) -> str:
    if lower == upper:
        return f"{lower}/{total} ({100 * lower / total:.1f}%)"
    return f"{lower}–{upper}/{total} ({100 * lower / total:.1f}–{100 * upper / total:.1f}%)"


def _format_half_count(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def format_midpoint_noise(lower: int, upper: int, total: int) -> str:
    midpoint = (lower + upper) / 2
    half_range = (upper - lower) / 2
    return (
        f"{_format_half_count(midpoint)} ± {_format_half_count(half_range)}/{total} "
        f"({100 * midpoint / total:.1f}% ± {100 * half_range / total:.1f} pp)"
    )


def add_range_noise_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    lower = int(metrics["correct"])
    upper = lower + int(metrics["unresolved"])
    total = int(metrics["total"])
    return {
        "accuracy_half_range": (upper - lower) / (2 * total),
        "accuracy_midpoint": (lower + upper) / (2 * total),
        "correct_half_range": (upper - lower) / 2,
        "correct_midpoint": (lower + upper) / 2,
    }


def comparison_table(per_variant: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# MATH-500 source-mapped v4 comparison",
        "",
        "Ranges count every unresolved row wrong at the lower end and right at the upper end. Midpoint ± half-range treats that width as evaluation noise.",
        "",
        "| Variant | Native | V4 strict range | Midpoint ± eval noise | Resolved | Definitive | Cap hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in per_variant:
        total = int(item["total"])
        lower = int(item["correct"])
        upper = lower + int(item["unresolved"])
        lines.append(
            "| {variant} | {native} | {strict} | {midpoint} | {resolved}/{total} | "
            "{definitive}/{total} | {caps}/{total} ({cap_percent:.1f}%) |".format(
                variant=item["variant"],
                native=format_count_range(int(item["native_correct_count"]), int(item["native_correct_count"]), total),
                strict=format_count_range(lower, upper, total),
                midpoint=format_midpoint_noise(lower, upper, total),
                resolved=item["resolved"],
                total=total,
                definitive=item["definitive"],
                caps=item["cap_hit_count"],
                cap_percent=item["cap_hit_percent"],
            )
        )
    lines.extend(
        [
            "",
            "## Cap-hit sensitivity",
            "",
            "`Cap hits wrong` forces every length-capped response wrong; `cap hits right` forces every length-capped response right. Other unresolved rows still produce the displayed range.",
            "",
            "| Variant | Cap hits | Cap hits correct range | Midpoint ± eval noise | Correct if cap hits wrong | Midpoint ± eval noise | Correct if cap hits right | Midpoint ± eval noise |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in per_variant:
        total = int(item["total"])
        cap_count = int(item["cap_hit_count"])
        cap_lower = int(item["cap_hits_correct_lower_bound"])
        cap_upper = int(item["cap_hits_correct_upper_bound"])
        wrong_lower = int(item["correct_if_all_cap_hits_wrong_lower_bound"])
        wrong_upper = int(item["correct_if_all_cap_hits_wrong_upper_bound"])
        right_lower = int(item["correct_if_all_cap_hits_right_lower_bound"])
        right_upper = int(item["correct_if_all_cap_hits_right_upper_bound"])
        lines.append(
            "| {variant} | {caps}/{total} ({cap_percent:.1f}%) | {cap_correct} | "
            "{cap_midpoint} | {wrong} | {wrong_midpoint} | {right} | {right_midpoint} |".format(
                variant=item["variant"],
                caps=cap_count,
                total=total,
                cap_percent=item["cap_hit_percent"],
                cap_correct=format_count_range(cap_lower, cap_upper, cap_count),
                cap_midpoint=format_midpoint_noise(cap_lower, cap_upper, cap_count),
                wrong=format_count_range(wrong_lower, wrong_upper, total),
                wrong_midpoint=format_midpoint_noise(wrong_lower, wrong_upper, total),
                right=format_count_range(right_lower, right_upper, total),
                right_midpoint=format_midpoint_noise(right_lower, right_upper, total),
            )
        )
    lines.extend(
        [
            "",
            "Unresolved rows remain queued for versioned review by a stronger reasoning model over the trace or by a human.",
            "",
        ]
    )
    return "\n".join(lines)


def score(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    value = inventory(repo_root)
    paths = roots(repo_root)
    gate_rows = all_gate_rows(repo_root)
    gold_rows = native.controller.read_jsonl(paths["gold"])
    gold_by_id = {str(row["problem_id"]): row for row in gold_rows}
    if len(gold_rows) != 500 or len(gold_by_id) != 500:
        raise RuntimeError("v3 gold structure manifest must contain 500 rows")
    native_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for item in value["checkpoints"]:
        variant = str(item["variant"])
        output = native.controller.absolute_item_paths(repo_root, item)["output"]
        native_rows = native.controller.read_jsonl(output / "merged/predictions.jsonl")
        if len(native_rows) != 500:
            raise RuntimeError(f"native predictions are incomplete: {variant}")
        native_by_variant[variant] = {str(row["problem_id"]): row for row in native_rows}

    completed = repair_score_journal(paths["scores"], gate_rows)
    ordered_scores: list[dict[str, Any]] = []
    for gate in gate_rows:
        key = str(gate["row_key"])
        current = completed.get(key)
        if current is None:
            problem_id = str(gate["problem_id"])
            variant = str(gate["variant"])
            current = score_row(
                gate, gold_by_id[problem_id], native_by_variant[variant][problem_id]
            )
            append_fsynced(paths["scores"], current)
        ordered_scores.append(current)
    native.controller.atomic_text(paths["scores"], native.controller.jsonl_text(ordered_scores))

    gate_by_key = {str(row["row_key"]): row for row in gate_rows}
    source_by_identity: dict[str, dict[str, Any]] = {}
    for variant in variants(value):
        for shard_index in range(EXPECTED_SHARDS):
            for source in native.controller.read_jsonl(input_path(repo_root, variant, shard_index)):
                source_by_identity[source_key(source)] = source
    review_rows: list[dict[str, Any]] = []
    for row in ordered_scores:
        if row["correct"] is not None:
            continue
        gate = gate_by_key[str(row["gate_row_key"])]
        source = source_by_identity[source_key(gate)]
        gold = gold_by_id[str(row["problem_id"])]
        review_rows.append(
            {
                "artifact_schema_version": GATE_ARTIFACT_VERSION,
                "automatic_result": row,
                "correct_answer": gold["correct_answer"],
                "gate_unresolved": gate["unresolved"],
                "manual_resolution_schema": {
                    "correct": "boolean",
                    "method": "stronger_model_trace_reasoning | human",
                    "rationale": "string",
                    "reviewer": "string",
                    "source_sha256": canonical_sha(gate),
                },
                "problem": source["problem"],
                "problem_id": source["problem_id"],
                "response": source["response"],
                "variant": source["variant"],
            }
        )
    native.controller.atomic_text(paths["review"], native.controller.jsonl_text(review_rows))

    per_variant: list[dict[str, Any]] = []
    for item in value["checkpoints"]:
        variant = str(item["variant"])
        rows = [row for row in ordered_scores if row["variant"] == variant]
        summary_metrics = v2.metrics(rows)
        native_correct_count = sum(row["native_correct"] for row in rows)
        summary = {
            "artifact_schema_version": GATE_ARTIFACT_VERSION,
            "benchmark": "HuggingFaceH4/MATH-500",
            "dataset_revision": value["dataset"]["revision"],
            "metric": METRIC_NAME,
            "native_correct_count": native_correct_count,
            "per_problem": rows,
            "scorer_version": SCORER_VERSION,
            "variant": variant,
            **summary_metrics,
            **add_range_noise_metrics(summary_metrics),
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
                "native_correct_count": native_correct_count,
                **summary_metrics,
                **add_range_noise_metrics(summary_metrics),
            }
        )

    suite_metrics = v2.metrics(ordered_scores)
    suite = {
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "completed_at": native.controller.now_iso(),
        "metric": METRIC_NAME,
        "results": per_variant,
        "scorer_version": SCORER_VERSION,
        "status": "complete" if suite_metrics["unresolved"] == 0 else "incomplete",
        **suite_metrics,
        **add_range_noise_metrics(suite_metrics),
    }
    native.controller.atomic_text(paths["suite_result"], native.controller.canonical_json(suite))
    native.controller.atomic_text(paths["table"], comparison_table(per_variant))
    print(
        f"LLAMA_V4_SCORE_PUBLISHED status={suite['status']} correct={suite['correct']}/"
        f"{suite['total']} unresolved={suite['unresolved']} review={len(review_rows)}",
        flush=True,
    )


def check_runtime(_args: argparse.Namespace) -> None:
    expected = policy()["runtime"]
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"v4 runtime mismatch: {actual} != {expected}")
    print("LLAMA_V4_RUNTIME_VALID " + " ".join(f"{k}={v}" for k, v in actual.items()))


def check_model(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    ready = v2.validate_model(repo_root)
    if ready.get("revision") != policy()["model"]["revision"]:
        raise RuntimeError("v4 model revision mismatch")
    print(f"LLAMA_V4_MODEL_VALID revision={ready['revision']}", flush=True)


def audit(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    value = inventory(repo_root)
    paths = roots(repo_root)
    metadata = native.controller.load_json(paths["gate_metadata"])
    if metadata.get("rows") != EXPECTED_ROWS or metadata.get("shards") != 48:
        raise RuntimeError("v3 metadata is incomplete")
    expected = policy()["one_off_reuse"]
    imported = metadata.get("one_off_reuse", {})
    for field in (
        "expected_adjudications",
        "expected_definitive",
        "expected_missing_adjudications",
        "expected_nondefinitive",
    ):
        if imported.get(field) != expected[field]:
            raise RuntimeError(f"v3 reuse audit mismatch: {field}")

    gate_rows = all_gate_rows(repo_root)
    input_by_source: dict[str, dict[str, Any]] = {}
    for variant in variants(value):
        for shard_index in range(EXPECTED_SHARDS):
            sources, _ = validated_inputs(repo_root, variant, shard_index)
            input_by_source.update({source_key(source): source for source in sources})
    extracted = 0
    for gate in gate_rows:
        source = input_by_source[source_key(gate)]
        if gate["adjudication_source"].get("artifact") != "strict-v2-one-off-reuse":
            raise RuntimeError("v3 current-corpus row lacks one-off reuse provenance")
        extraction = gate.get("extraction")
        if extraction is None:
            continue
        extracted += 1
        if any(field in extraction for field in ("candidate_id", "start_boundary_id", "end_boundary_id")):
            raise RuntimeError("v3 extraction retained an identifier field")
        regions = v2.response_regions(str(source["response"]))
        region = {region["id"]: region for region in regions}[
            gate["adjudication"]["evidence_region_id"]
        ]
        boxed, heuristic = eligible_candidates(str(source["response"]), region)
        validated = selected_extraction(
            {
                "selection_mode": extraction["requested_selection_mode"],
                "final_answer_raw": extraction["generated_answer_raw"],
            },
            str(source["response"]),
            region,
            boxed,
            heuristic,
        )
        if canonical_sha(validated) != canonical_sha(extraction):
            raise RuntimeError("v4 extraction is not deterministic under audit")

    score_rows = native.controller.read_jsonl(paths["scores"])
    if len(score_rows) != EXPECTED_ROWS:
        raise RuntimeError("v3 score journal is not 6000 rows")
    suite = native.controller.load_json(paths["suite_result"])
    if suite.get("total") != EXPECTED_ROWS or suite.get("status") not in {"complete", "incomplete"}:
        raise RuntimeError("v3 suite result has invalid publication state")
    if suite["status"] == "complete" and suite["unresolved"] != 0:
        raise RuntimeError("complete v3 suite still has unresolved rows")
    if suite["status"] == "incomplete" and not suite["unresolved"]:
        raise RuntimeError("incomplete v3 suite has no unresolved rows")
    for item in value["checkpoints"]:
        result = native.controller.load_json(result_path(repo_root, item) / "RESULTS.json")
        if result.get("total") != 500 or len(result.get("per_problem", [])) != 500:
            raise RuntimeError(f"incomplete v3 result: {item['variant']}")
        for field in (
            "accuracy_lower_bound",
            "accuracy_upper_bound",
            "cap_hit_percent",
            "cap_hits_correct_lower_bound",
            "cap_hits_correct_upper_bound",
            "correct_if_all_cap_hits_wrong_lower_bound",
            "correct_if_all_cap_hits_wrong_upper_bound",
            "correct_if_all_cap_hits_right_lower_bound",
            "correct_if_all_cap_hits_right_upper_bound",
        ):
            if field not in result:
                raise RuntimeError(f"v3 result lacks {field}: {item['variant']}")
    audit_value = {
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "audited_at": native.controller.now_iso(),
        "extractions": extracted,
        "gate_rows": len(gate_rows),
        "metric": METRIC_NAME,
        "score_rows": len(score_rows),
        "status": suite["status"],
        "unresolved": suite["unresolved"],
        "variants": EXPECTED_VARIANTS,
    }
    native.controller.atomic_text(paths["audit"], native.controller.canonical_json(audit_value))
    print(
        f"LLAMA_V3_AUDIT_PUBLISHED status={suite['status']} rows={EXPECTED_ROWS} "
        f"extractions={extracted} unresolved={suite['unresolved']}",
        flush=True,
    )


def record_submission(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    value = inventory(repo_root)
    assignments: dict[str, str] = {}
    for assignment in args.extraction_array_job:
        variant, separator, job_id = assignment.partition("=")
        if not separator or variant in assignments or variant not in variants(value) or not job_id.isdigit():
            raise RuntimeError(f"invalid v3 extraction array assignment: {assignment}")
        assignments[variant] = job_id
    if set(assignments) != set(variants(value)):
        raise RuntimeError("v3 submission must record all twelve extraction arrays")
    for job_id in (args.prepare_job, args.canary_job, args.score_job, args.audit_job):
        if not str(job_id).isdigit():
            raise RuntimeError(f"invalid v3 Slurm job ID: {job_id}")
    destination = roots(repo_root)["gate"] / "submission.json"
    if destination.exists():
        raise RuntimeError(f"refusing to replace v3 submission journal: {destination}")
    submission = {
        "artifact_schema_version": GATE_ARTIFACT_VERSION,
        "audit_job": str(args.audit_job),
        "canary_job": str(args.canary_job),
        "extraction_array_jobs": assignments,
        "extraction_shards": 48,
        "model": policy()["model"],
        "one_off_reuse": policy()["one_off_reuse"],
        "policy_sha256": gate_policy_sha256(),
        "prepare_job": str(args.prepare_job),
        "recorded_at": native.controller.now_iso(),
        "score_job": str(args.score_job),
    }
    native.controller.atomic_text(destination, native.controller.canonical_json(submission))
    print(f"LLAMA_V3_SUBMISSION_RECORDED path={destination}", flush=True)


def show_path(args: argparse.Namespace) -> None:
    repo_root = native.controller.repo_root_from_args(args)
    available = roots(repo_root)
    if args.field == "input":
        if args.variant is None or args.shard_index is None:
            raise RuntimeError("v3 input path requires variant and shard")
        result = input_path(repo_root, args.variant, args.shard_index)
    elif args.field == "decision":
        if args.variant is None or args.shard_index is None:
            raise RuntimeError("v3 decision path requires variant and shard")
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

    replay_parser = subparsers.add_parser("replay-v3-and-score")
    replay_parser.set_defaults(func=replay_v3_suite)

    extraction_parser = subparsers.add_parser("run-extraction")
    extraction_parser.add_argument("--variant", required=True)
    extraction_parser.add_argument("--shard-index", required=True, type=int)
    extraction_parser.set_defaults(func=run_extraction)

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
            "audit",
            "canary_complete",
            "canary_inputs",
            "canary_outputs",
            "decision",
            "gate",
            "gate_inputs",
            "gate_metadata",
            "gate_outputs",
            "gold",
            "input",
            "model",
            "python",
            "review",
            "runtime",
            "scores",
            "scoring",
            "suite_result",
            "table",
        ),
        required=True,
    )
    path_parser.add_argument("--variant")
    path_parser.add_argument("--shard-index", type=int)
    path_parser.set_defaults(func=show_path)

    submission_parser = subparsers.add_parser("record-submission")
    submission_parser.add_argument("--prepare-job", required=True)
    submission_parser.add_argument("--canary-job", required=True)
    submission_parser.add_argument("--extraction-array-job", action="append", default=[])
    submission_parser.add_argument("--score-job", required=True)
    submission_parser.add_argument("--audit-job", required=True)
    submission_parser.set_defaults(func=record_submission)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
