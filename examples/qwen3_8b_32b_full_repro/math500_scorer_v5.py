#!/usr/bin/env python3
"""Gold-independent, last-boxed-answer scorer for MATH-500.

The model answer is extracted only from the chronologically last complete
``\\boxed{...}`` group.  Candidate and gold are then interpreted by the same
fixed parser ensemble; no parser or answer type is selected from the gold.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from examples.qwen3_8b_32b_full_repro.math500_scorer_v4 import (
    normalize_answer_units,
)


SCORER_VERSION = "math500_last_boxed_symmetric_scorer_v5"
TRACE_SCHEMA_VERSION = "math500_last_boxed_symmetric_trace_v5"
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
BOX_START_RE = re.compile(r"\\boxed\s*\{")
ADJACENT_BOX_GAP_RE = re.compile(
    r"(?is)^(?:\s|[,;:&/]|and\b|or\b|\\quad\b|\\qquad\b|\\[,;:]|\\\\)*$"
)
MCQ_RE = re.compile(r"(?is)^\s*(?:\\(?:text|mathrm|mbox)\s*\{\s*)?\(?\s*([A-E])\s*\)?\s*\}?\s*$")
TEXT_WRAPPER_RE = re.compile(
    r"(?is)^\s*\\(?:text|mathrm|mbox)\s*\{(?P<value>[^{}]*)\}\s*$"
)
PURE_TEXT_RE = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+)*$")
MATRIX_RE = re.compile(
    r"(?s)^\s*\\begin\{(?P<kind>[bBpPvV]?matrix|smallmatrix)\}.*"
    r"\\end\{(?P=kind)\}\s*$"
)
RELATION_RE = re.compile(
    r"(?:\\(?:leq?|geq?|neq|in)|<=|>=|!=|(?<![<>=])=(?!=)|(?<!\\)[<>])"
)
THOUSANDS_NUMBER_RE = re.compile(
    r"(?x)^\s*[+-]?\d{1,3}(?:,\s*(?:\\!\s*)?\d{3})+(?:\.\d+)?\s*$"
)


@dataclass(frozen=True)
class BoxEvent:
    event_id: str
    span: tuple[int, int]
    raw: str
    interior_raw: str | None
    complete: bool
    valid: bool
    invalidation_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoxGroup:
    group_id: str
    span: tuple[int, int]
    raw_source: str
    component_event_ids: tuple[str, ...]
    components_raw: tuple[str, ...]


@dataclass(frozen=True)
class Interpretation:
    kind: str
    value: str
    components: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _sha256(value: str | None) -> str:
    return hashlib.sha256(str(value or "").encode()).hexdigest()


def clean_generated_text(text: str | None) -> str:
    result = str(text or "")
    for token in SPECIAL_TOKENS:
        result = result.replace(token, "")
    return result


def _is_escaped(text: str, position: int) -> bool:
    count = 0
    cursor = position - 1
    while cursor >= 0 and text[cursor] == "\\":
        count += 1
        cursor -= 1
    return bool(count % 2)


def extract_box_events(text: str) -> tuple[list[BoxEvent], list[dict[str, Any]]]:
    events: list[BoxEvent] = []
    malformed: list[dict[str, Any]] = []
    for match in BOX_START_RE.finditer(text):
        cursor = match.end()
        depth = 1
        while cursor < len(text) and depth:
            if text[cursor] == "{" and not _is_escaped(text, cursor):
                depth += 1
            elif text[cursor] == "}" and not _is_escaped(text, cursor):
                depth -= 1
            cursor += 1
        if depth:
            malformed.append(
                {
                    "span": [match.start(), len(text)],
                    "raw": text[match.start() :],
                    "reason": "unclosed_box",
                }
            )
            continue
        interior = text[match.end() : cursor - 1]
        reasons: list[str] = []
        if not interior.strip():
            reasons.append("empty_box")
        events.append(
            BoxEvent(
                event_id=f"box_{len(events):04d}",
                span=(match.start(), cursor),
                raw=text[match.start() : cursor],
                interior_raw=interior,
                complete=True,
                valid=not reasons,
                invalidation_reasons=tuple(reasons),
            )
        )
    return events, malformed


def group_box_events(text: str, events: Iterable[BoxEvent]) -> list[BoxGroup]:
    valid = [event for event in events if event.valid and event.interior_raw is not None]
    raw_groups: list[list[BoxEvent]] = []
    current: list[BoxEvent] = []
    for event in valid:
        if current:
            gap = text[current[-1].span[1] : event.span[0]]
            display_boundary = bool(re.search(r"\\\]\s*\\\[|\$\$\s*\$\$", gap))
            adjacent = len(gap) <= 80 and bool(ADJACENT_BOX_GAP_RE.fullmatch(gap))
            if display_boundary or not adjacent:
                raw_groups.append(current)
                current = []
        current.append(event)
    if current:
        raw_groups.append(current)
    result: list[BoxGroup] = []
    for members in raw_groups:
        start, end = members[0].span[0], members[-1].span[1]
        result.append(
            BoxGroup(
                group_id=f"boxed_group_{len(result):04d}",
                span=(start, end),
                raw_source=text[start:end],
                component_event_ids=tuple(member.event_id for member in members),
                components_raw=tuple(str(member.interior_raw) for member in members),
            )
        )
    return result


def extract_last_boxed_group(
    text: str,
) -> tuple[BoxGroup | None, list[BoxEvent], list[BoxGroup], list[dict[str, Any]]]:
    events, malformed = extract_box_events(text)
    groups = group_box_events(text, events)
    return (groups[-1] if groups else None), events, groups, malformed


def _strip_thousands(value: str) -> str:
    # A comma is removed only when the entire value is an explicitly grouped
    # numeric literal.  This keeps ``(12,102)`` as a two-endpoint interval
    # while still normalizing ``58,500`` and ``11,\!111,\!111``.
    if THOUSANDS_NUMBER_RE.fullmatch(value):
        value = re.sub(r",\s*(?:\\!\s*)?", "", value)
    return value


def _strip_left_right(value: str) -> str:
    return re.sub(r"\\(?:left|right)\s*", "", value)


def _top_level_parts(value: str, separators: tuple[str, ...]) -> list[str]:
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    cursor = 0
    while cursor < len(value):
        char = value[cursor]
        if char in pairs and not _is_escaped(value, cursor):
            stack.append(pairs[char])
            cursor += 1
            continue
        if stack and char == stack[-1] and not _is_escaped(value, cursor):
            stack.pop()
            cursor += 1
            continue
        if not stack:
            matched = next(
                (token for token in separators if value.startswith(token, cursor)),
                None,
            )
            if matched is not None:
                parts.append(value[start:cursor].strip())
                cursor += len(matched)
                start = cursor
                continue
        cursor += 1
    parts.append(value[start:].strip())
    return parts


def _split_conjunction(value: str) -> list[str]:
    comma = _top_level_parts(value, (",",))
    if len(comma) > 1:
        return [part for part in comma if part]
    # Conjunctions are accepted only as standalone top-level words.
    match = re.search(r"(?i)\s+(?:and|or)\s+", value)
    if not match:
        return [value]
    left = value[: match.start()].strip()
    right = value[match.end() :].strip()
    return [left, right] if left and right else [value]


def _expand_plus_minus_parts(parts: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for part in parts:
        if part.count(r"\pm") == 1:
            expanded.extend(
                [part.replace(r"\pm", "+"), part.replace(r"\pm", "-")]
            )
        else:
            expanded.append(part)
    return expanded


def _unwrap_text(value: str) -> str:
    match = TEXT_WRAPPER_RE.fullmatch(value)
    return match.group("value") if match else value.strip()


def _categorical(value: str) -> Interpretation | None:
    match = MCQ_RE.fullmatch(value)
    if match:
        return Interpretation("multiple_choice", match.group(1).upper())
    plain = re.sub(r"\s+", " ", _unwrap_text(value).strip())
    if PURE_TEXT_RE.fullmatch(plain) and len(plain.replace(" ", "")) > 1:
        return Interpretation("categorical_text", plain.casefold())
    return None


def _outer_sequence(value: str) -> tuple[str, str, list[str]] | None:
    if len(value) < 2:
        return None
    opening, closing = value[0], value[-1]
    if opening not in "([" or closing not in ")]":
        return None
    inner = value[1:-1]
    parts = _top_level_parts(inner, (",",))
    if len(parts) < 2 or any(not part for part in parts):
        return None
    return opening, closing, parts


def _single_interpretations(raw_value: str) -> list[Interpretation]:
    unit = normalize_answer_units(raw_value)
    value = _strip_thousands(_strip_left_right(str(unit["normalized"]).strip()))
    if not value:
        return []
    categorical = _categorical(value)
    if categorical is not None:
        return [categorical]
    if MATRIX_RE.fullmatch(value):
        return [Interpretation("matrix", value)]
    union_parts = _top_level_parts(value, (r"\cup",))
    if len(union_parts) > 1 and all(union_parts):
        return [
            Interpretation(
                "interval_union",
                value,
                tuple(union_parts),
                {"component_count": len(union_parts)},
            )
        ]
    if (value.startswith(r"\{") and value.endswith(r"\}")) or (
        value.startswith("{") and value.endswith("}")
    ):
        left = 2 if value.startswith(r"\{") else 1
        right = -2 if value.endswith(r"\}") else -1
        parts = _expand_plus_minus_parts(
            part for part in _top_level_parts(value[left:right], (",",)) if part
        )
        return [
            Interpretation(
                "finite_set",
                value,
                tuple(parts),
                {"cardinality": len(parts)},
            )
        ]
    sequence = _outer_sequence(value)
    if sequence is not None:
        opening, closing, parts = sequence
        result: list[Interpretation] = []
        if opening == "(" and closing == ")":
            result.append(
                Interpretation(
                    "ordered_tuple",
                    value,
                    tuple(parts),
                    {"arity": len(parts)},
                )
            )
        if len(parts) == 2:
            result.append(
                Interpretation(
                    "interval",
                    value,
                    tuple(parts),
                    {
                        "left_closed": opening == "[",
                        "right_closed": closing == "]",
                    },
                )
            )
        return result
    components = _expand_plus_minus_parts(_split_conjunction(value))
    if len(components) > 1:
        return [
            Interpretation(
                "unordered_answers",
                value,
                tuple(components),
                {"cardinality": len(components), "source": "top_level_separator"},
            )
        ]
    if RELATION_RE.search(value):
        return [Interpretation("relation", value)]
    return [Interpretation("expression", value)]


def interpret_answer(
    components_raw: tuple[str, ...] | list[str] | str,
) -> list[Interpretation]:
    if isinstance(components_raw, str):
        components = (components_raw,)
    else:
        components = tuple(components_raw)
    if len(components) > 1:
        normalized = tuple(
            _strip_thousands(
                _strip_left_right(str(normalize_answer_units(part)["normalized"]).strip())
            )
            for part in components
        )
        normalized = tuple(_expand_plus_minus_parts(normalized))
        return [
            Interpretation(
                "unordered_answers",
                ", ".join(normalized),
                normalized,
                {"cardinality": len(normalized), "source": "adjacent_boxes"},
            )
        ]
    return _single_interpretations(components[0])


def _semantic_class(value: Any) -> str:
    from sympy import Tuple
    from sympy.core.relational import Relational
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.sets.sets import Set

    if isinstance(value, Relational):
        return "relation"
    if isinstance(value, MatrixBase):
        return "matrix"
    if isinstance(value, Tuple):
        return "tuple"
    if isinstance(value, Set):
        return "set"
    return "expression"


@lru_cache(maxsize=65536)
def _parse_math_lanes(value: str) -> tuple[tuple[str, str, Any], ...]:
    from math_verify import parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    lanes = (
        ("latex", f"${value}$", LatexExtractionConfig()),
        ("expr", value, ExprExtractionConfig()),
    )
    parsed: list[tuple[str, str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for backend, source, config in lanes:
        try:
            objects = parse(
                source,
                extraction_config=[config],
                fallback_mode="no_fallback",
                extraction_mode="first_match",
                parsing_timeout=5,
                raise_on_error=True,
            )
        except Exception:
            continue
        if not isinstance(objects, list) or len(objects) != 1:
            continue
        obj = objects[0]
        semantic_class = _semantic_class(obj)
        key = (semantic_class, repr(obj))
        if key not in seen:
            seen.add(key)
            parsed.append((backend, semantic_class, obj))
    return tuple(parsed)


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
            if c_class != g_class or (required_class is not None and g_class != required_class):
                continue
            try:
                equivalent = bool(
                    verify(
                        g_obj,
                        c_obj,
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
                    "gold_backend": g_backend,
                    "gold_class": g_class,
                    "equivalent": equivalent,
                    "error": error,
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
                (item for item in _single_interpretations(value) if item.kind == "interval"),
                None,
            )
            for value in candidate.components
        ]
        gold_intervals = [
            next(
                (item for item in _single_interpretations(value) if item.kind == "interval"),
                None,
            )
            for value in gold.components
        ]
        if any(item is None for item in candidate_intervals + gold_intervals):
            return False, {"cardinality_match": True, "interval_parse_failure": True}
        edges: list[list[bool]] = []
        for candidate_interval in candidate_intervals:
            edges.append(
                [
                    compare_interpretations(candidate_interval, gold_interval)[0]
                    for gold_interval in gold_intervals
                ]
            )

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
                    for item in _single_interpretations(component)
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
                    "candidate_interpretations": [asdict(item) for item in candidate_interpretations],
                    "gold_interpretations": [asdict(item) for item in gold_interpretations],
                    "attempts": attempts,
                }
    return {
        "parse_success": parse_success,
        "equivalent": False,
        "candidate_interpretations": [asdict(item) for item in candidate_interpretations],
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
    text = clean_generated_text(generated_text)
    selected, events, groups, malformed = extract_last_boxed_group(text)
    if selected is None:
        verification = {
            "parse_success": False,
            "equivalent": False,
            "candidate_interpretations": [],
            "gold_interpretations": [asdict(item) for item in interpret_answer(gold_answer)],
            "attempts": [],
        }
        reason = "no_valid_complete_box"
        correct = False
        scorable = False
    else:
        verification = verify_symmetric(selected.components_raw, str(gold_answer))
        scorable = bool(verification["parse_success"])
        correct = bool(scorable and verification["equivalent"])
        reason = "correct" if correct else ("incorrect_math" if scorable else "parse_failure")
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
        "official_candidate_sha256": _sha256(json.dumps(selected_raw, ensure_ascii=False)),
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
        "generated_text_sha256": _sha256(generated_text),
        "gold_answer_sha256": _sha256(gold_answer),
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
