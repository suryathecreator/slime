#!/usr/bin/env python3
"""Strict last-box MATH-500 scorer with whole-value mathematical parsing."""

from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any

from examples.qwen3_8b_32b_full_repro import math500_scorer_v5 as v5


SCORER_VERSION = "math500_last_boxed_symmetric_scorer_v6"
TRACE_SCHEMA_VERSION = "math500_last_boxed_symmetric_trace_v6"

Interpretation = v5.Interpretation

DECIMAL_LITERAL_RE = re.compile(
    r"(?<![\d.])(?P<whole>\d*)\.(?P<fraction>\d+)(?![\d.])"
)
PERCENT_SUFFIX_RE = re.compile(r"(?s)(?:\\%|%)\s*$")


def _strip_terminal_percent(value: str) -> str:
    """Remove a terminal presentation percent sign from a verification copy."""

    match = PERCENT_SUFFIX_RE.search(value)
    return value[: match.start()].rstrip() if match else value


def _exact_decimal_latex(value: str) -> str:
    """Rewrite decimal literals as exact fractions before LaTeX parsing."""

    def replace(match: re.Match[str]) -> str:
        fraction = match.group("fraction")
        digits = (match.group("whole") or "0") + fraction
        return rf"\frac{{{int(digits)}}}{{{10 ** len(fraction)}}}"

    return DECIMAL_LITERAL_RE.sub(replace, value)


def _prepare_components(
    components_raw: tuple[str, ...] | list[str] | str,
) -> tuple[str, ...] | str:
    if isinstance(components_raw, str):
        return _strip_terminal_percent(components_raw)
    return tuple(_strip_terminal_percent(value) for value in components_raw)


def interpret_answer(
    components_raw: tuple[str, ...] | list[str] | str,
) -> list[Interpretation]:
    """Interpret candidate and gold through the same fixed, gold-independent path."""

    return v5.interpret_answer(_prepare_components(components_raw))


def _parse_math_lanes(value: str) -> tuple[tuple[str, str, Any], ...]:
    """Parse the complete value only through an explicitly wrapped LaTeX lane."""

    from math_verify import parse
    from math_verify.parser import LatexExtractionConfig

    verification_value = _exact_decimal_latex(value)
    try:
        objects = parse(
            f"${verification_value}$",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=5,
            raise_on_error=True,
        )
    except Exception:
        return ()
    if not isinstance(objects, list) or len(objects) != 1:
        return ()
    parsed = objects[0]
    return (("latex", v5._semantic_class(parsed), parsed),)


def _required_semantic_class(kind: str) -> str | None:
    return {
        "expression": "expression",
        "relation": "relation",
        "matrix": "matrix",
    }.get(kind)


def _math_equivalent(
    candidate: str,
    gold: str,
    *,
    required_class: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    from math_verify import verify

    candidate_parsed = _parse_math_lanes(candidate)
    gold_parsed = _parse_math_lanes(gold)
    attempts: list[dict[str, Any]] = []
    for c_backend, c_class, c_obj in candidate_parsed:
        if required_class is not None and c_class != required_class:
            continue
        for g_backend, g_class, g_obj in gold_parsed:
            if c_class != g_class or (
                required_class is not None and g_class != required_class
            ):
                continue
            try:
                equivalent = bool(
                    verify(
                        g_obj,
                        c_obj,
                        float_rounding=15,
                        numeric_precision=30,
                        strict=True,
                        allow_set_relation_comp=False,
                        timeout_seconds=5,
                        raise_on_error=True,
                    )
                )
                error = None
            except Exception as exc:
                equivalent = False
                error = f"{type(exc).__name__}: {exc}"
            attempts.append(
                {
                    "candidate_backend": c_backend,
                    "candidate_class": c_class,
                    "candidate_parsed_repr": repr(c_obj),
                    "equivalent": equivalent,
                    "error": error,
                    "gold_backend": g_backend,
                    "gold_class": g_class,
                    "gold_parsed_repr": repr(g_obj),
                }
            )
            if equivalent:
                return True, {
                    "candidate_parse_count": len(candidate_parsed),
                    "gold_parse_count": len(gold_parsed),
                    "attempts": attempts,
                }
    return False, {
        "candidate_parse_count": len(candidate_parsed),
        "gold_parse_count": len(gold_parsed),
        "attempts": attempts,
    }


def _ordered_components(
    candidate: tuple[str, ...], gold: tuple[str, ...]
) -> tuple[bool, dict[str, Any]]:
    if len(candidate) != len(gold):
        return False, {"cardinality_match": False, "component_results": []}
    results: list[dict[str, Any]] = []
    for candidate_part, gold_part in zip(candidate, gold, strict=True):
        equivalent, evidence = _math_equivalent(candidate_part, gold_part)
        results.append({"equivalent": equivalent, "evidence": evidence})
        if not equivalent:
            return False, {"cardinality_match": True, "component_results": results}
    return True, {"cardinality_match": True, "component_results": results}


def _unordered_components(
    candidate: tuple[str, ...], gold: tuple[str, ...]
) -> tuple[bool, dict[str, Any]]:
    if len(candidate) != len(gold):
        return False, {"cardinality_match": False, "matches": []}
    edges: list[list[bool]] = []
    evidence: list[list[dict[str, Any]]] = []
    for candidate_part in candidate:
        edge_row: list[bool] = []
        evidence_row: list[dict[str, Any]] = []
        for gold_part in gold:
            equivalent, detail = _math_equivalent(candidate_part, gold_part)
            edge_row.append(equivalent)
            evidence_row.append(detail)
        edges.append(edge_row)
        evidence.append(evidence_row)

    def match(index: int, used: frozenset[int]) -> tuple[int, ...] | None:
        if index == len(candidate):
            return ()
        for gold_index, equivalent in enumerate(edges[index]):
            if equivalent and gold_index not in used:
                tail = match(index + 1, used | {gold_index})
                if tail is not None:
                    return (gold_index,) + tail
        return None

    assignment = match(0, frozenset())
    return assignment is not None, {
        "cardinality_match": True,
        "assignment": list(assignment) if assignment is not None else None,
        "edge_evidence": evidence,
    }


def compare_interpretations(
    candidate: Interpretation, gold: Interpretation
) -> tuple[bool, dict[str, Any]]:
    if candidate.kind != gold.kind:
        return False, {"reason": "type_mismatch"}
    if candidate.kind in {"categorical_text", "multiple_choice"}:
        equivalent = candidate.value == gold.value
        return equivalent, {"exact_normalized_text_match": equivalent}
    if candidate.kind in {"expression", "relation", "matrix"}:
        equivalent, evidence = _math_equivalent(
            candidate.value,
            gold.value,
            required_class=_required_semantic_class(candidate.kind),
        )
        if equivalent and candidate.kind == "matrix":
            candidate_shapes = {
                obj.shape
                for _, semantic_class, obj in _parse_math_lanes(candidate.value)
                if semantic_class == "matrix"
            }
            gold_shapes = {
                obj.shape
                for _, semantic_class, obj in _parse_math_lanes(gold.value)
                if semantic_class == "matrix"
            }
            shape_match = bool(candidate_shapes & gold_shapes)
            evidence["shape_match"] = shape_match
            equivalent = equivalent and shape_match
        return equivalent, evidence
    if candidate.kind == "ordered_tuple":
        return _ordered_components(candidate.components, gold.components)
    if candidate.kind == "interval":
        boundary_match = candidate.metadata == gold.metadata
        equivalent, evidence = _ordered_components(candidate.components, gold.components)
        evidence["boundary_match"] = boundary_match
        return equivalent and boundary_match, evidence
    if candidate.kind in {"finite_set", "unordered_answers"}:
        return _unordered_components(candidate.components, gold.components)
    if candidate.kind == "interval_union":
        if len(candidate.components) != len(gold.components):
            return False, {"cardinality_match": False}
        candidate_intervals = [
            next(
                (
                    item
                    for item in interpret_answer(value)
                    if item.kind == "interval"
                ),
                None,
            )
            for value in candidate.components
        ]
        gold_intervals = [
            next(
                (
                    item
                    for item in interpret_answer(value)
                    if item.kind == "interval"
                ),
                None,
            )
            for value in gold.components
        ]
        if any(item is None for item in candidate_intervals + gold_intervals):
            return False, {"cardinality_match": True, "interval_parse_failure": True}
        edges = [
            [
                compare_interpretations(candidate_interval, gold_interval)[0]
                for gold_interval in gold_intervals
            ]
            for candidate_interval in candidate_intervals
        ]

        def match(index: int, used: frozenset[int]) -> bool:
            if index == len(edges):
                return True
            return any(
                equivalent
                and gold_index not in used
                and match(index + 1, used | {gold_index})
                for gold_index, equivalent in enumerate(edges[index])
            )

        equivalent = match(0, frozenset())
        return equivalent, {"cardinality_match": True, "edges": edges}
    return False, {"reason": "unsupported_interpretation"}


def _interpretation_parseable(value: Interpretation) -> bool:
    if value.kind in {"categorical_text", "multiple_choice"}:
        return True
    if value.kind in {"expression", "relation", "matrix"}:
        required = _required_semantic_class(value.kind)
        return any(
            required is None or semantic_class == required
            for _, semantic_class, _ in _parse_math_lanes(value.value)
        )
    if value.kind in {
        "ordered_tuple",
        "interval",
        "finite_set",
        "unordered_answers",
    }:
        return bool(value.components) and all(
            _parse_math_lanes(component) for component in value.components
        )
    if value.kind == "interval_union":
        intervals = [
            next(
                (
                    item
                    for item in interpret_answer(component)
                    if item.kind == "interval"
                ),
                None,
            )
            for component in value.components
        ]
        return bool(intervals) and all(
            interval is not None and _interpretation_parseable(interval)
            for interval in intervals
        )
    return False


def verify_symmetric(
    candidate_components: tuple[str, ...], gold_answer: str
) -> dict[str, Any]:
    candidate_interpretations = interpret_answer(candidate_components)
    gold_interpretations = interpret_answer(gold_answer)
    attempts: list[dict[str, Any]] = []
    parse_success = any(
        candidate.kind == gold.kind
        and _interpretation_parseable(candidate)
        and _interpretation_parseable(gold)
        for candidate in candidate_interpretations
        for gold in gold_interpretations
    )
    for candidate in candidate_interpretations:
        for gold in gold_interpretations:
            if candidate.kind != gold.kind:
                continue
            equivalent, evidence = compare_interpretations(candidate, gold)
            attempts.append(
                {
                    "candidate_kind": candidate.kind,
                    "gold_kind": gold.kind,
                    "equivalent": equivalent,
                    "evidence": evidence,
                }
            )
            if equivalent:
                return {
                    "parse_success": True,
                    "equivalent": True,
                    "candidate_interpretations": [
                        asdict(item) for item in candidate_interpretations
                    ],
                    "gold_interpretations": [
                        asdict(item) for item in gold_interpretations
                    ],
                    "attempts": attempts,
                }
    return {
        "parse_success": parse_success,
        "equivalent": False,
        "candidate_interpretations": [
            asdict(item) for item in candidate_interpretations
        ],
        "gold_interpretations": [asdict(item) for item in gold_interpretations],
        "attempts": attempts,
    }


def score_response(
    generated_text: str,
    gold_answer: str,
    *,
    finish_reason: str | None,
    stop_token_id: int | None,
    cap_hit: bool,
    problem_id: str | int,
) -> dict[str, Any]:
    text = v5.clean_generated_text(generated_text)
    selected, events, groups, malformed = v5.extract_last_boxed_group(text)
    if selected is None:
        verification = {
            "parse_success": False,
            "equivalent": False,
            "candidate_interpretations": [],
            "gold_interpretations": [
                asdict(item) for item in interpret_answer(gold_answer)
            ],
            "attempts": [],
        }
        reason = "no_valid_complete_box"
        correct = False
        scorable = False
    else:
        verification = verify_symmetric(selected.components_raw, str(gold_answer))
        scorable = bool(verification["parse_success"])
        correct = bool(scorable and verification["equivalent"])
        reason = "correct" if correct else (
            "incorrect_math" if scorable else "parse_failure"
        )
    selected_raw = list(selected.components_raw) if selected else []
    return {
        "scorer_version": SCORER_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "problem_id": str(problem_id),
        "is_scorable": scorable,
        "is_correct": correct,
        "official_decision_reason": reason,
        "official_extracted_components_raw": selected_raw,
        "official_extracted_source_span": list(selected.span) if selected else None,
        "official_extracted_source_raw": selected.raw_source if selected else None,
        "official_candidate_sha256": v5._sha256(
            json.dumps(selected_raw, ensure_ascii=False)
        ),
        "extraction_rule": (
            "last_adjacent_boxed_group"
            if selected and len(selected_raw) > 1
            else "last_complete_box"
            if selected
            else "no_complete_box"
        ),
        "box_events": [asdict(event) for event in events],
        "boxed_groups": [asdict(group) for group in groups],
        "malformed_boxes": malformed,
        "selected_box_group_id": selected.group_id if selected else None,
        "typed_verifier_evidence": verification,
        "finish_reason": finish_reason,
        "stop_token_observed": stop_token_id,
        "cap_status": bool(cap_hit),
        "generated_text_sha256": v5._sha256(generated_text),
        "gold_answer_sha256": v5._sha256(gold_answer),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--problem-id", default="manual")
    parser.add_argument("--cap-hit", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            score_response(
                args.response,
                args.gold,
                finish_reason="length" if args.cap_hit else "stop",
                stop_token_id=None,
                cap_hit=args.cap_hit,
                problem_id=args.problem_id,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
