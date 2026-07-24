#!/usr/bin/env python3
"""Strict boxed-answer MATH-500 scorer with cap-hit diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from examples.qwen3_8b_32b_full_repro import math500_scorer_v1 as v1
from examples.qwen3_8b_32b_full_repro import math500_scorer_v2 as v2


SCORER_VERSION = "math500_strict_boxed_scorer_v3"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BOX_START_RE = re.compile(r"\\boxed\s*\{")
ADJACENT_BOX_GAP_RE = re.compile(
    r"(?is)^(?:\s|[,;:&/]|and\b|or\b|\\quad\b|\\qquad\b|\\[,;:]|\\\\)*$"
)


@dataclass
class BoxEvent:
    event_id: str
    span: list[int]
    raw: str
    candidate: str | None
    complete: bool
    structurally_valid: bool
    candidate_sha256: str
    invalidation_reasons: list[str] = field(default_factory=list)


@dataclass
class BoxGroup:
    event_id: str
    span: list[int]
    raw: str
    candidate: str
    components: list[str]
    source_event_ids: list[str]
    candidate_sha256: str


def clean_text(text: str | None) -> str:
    return v1.clean_text(text)


def _sha256(value: str | None) -> str:
    return hashlib.sha256(str(value or "").encode()).hexdigest()


def _is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    cursor = position - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return bool(backslashes % 2)


def classify_think_tags(text: str) -> dict[str, Any]:
    open_spans = [[match.start(), match.end()] for match in re.finditer(re.escape(THINK_OPEN), text)]
    close_spans = [[match.start(), match.end()] for match in re.finditer(re.escape(THINK_CLOSE), text)]
    if not open_spans and not close_spans:
        state = "none"
    elif len(open_spans) == 1 and len(close_spans) == 1 and open_spans[0][0] < close_spans[0][0]:
        state = "complete"
    elif open_spans and not close_spans:
        state = "open_without_close"
    elif close_spans and not open_spans:
        state = "close_without_open"
    else:
        state = "malformed_or_repeated"
    return {
        "state": state,
        "open_count": len(open_spans),
        "close_count": len(close_spans),
        "open_spans": open_spans,
        "close_spans": close_spans,
    }


def answer_region(text: str) -> tuple[str, int, str, dict[str, Any]]:
    think = classify_think_tags(text)
    return text, 0, "whole_response_all_think_states", think


def _balanced_boxes(text: str, offset: int) -> tuple[list[BoxEvent], list[dict[str, Any]]]:
    boxes: list[BoxEvent] = []
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
                    "span": [offset + match.start(), offset + len(text)],
                    "raw": text[match.start() :],
                    "reason": "unclosed_box",
                }
            )
            continue
        candidate = text[match.end() : cursor - 1].strip()
        reasons: list[str] = []
        if not candidate:
            reasons.append("empty_box")
        elif v2.is_structural_only(candidate):
            reasons.append("structural_only_box")
        boxes.append(
            BoxEvent(
                event_id=f"box_{len(boxes):04d}",
                span=[offset + match.start(), offset + cursor],
                raw=text[match.start() : cursor],
                candidate=candidate or None,
                complete=True,
                structurally_valid=not reasons,
                candidate_sha256=_sha256(candidate),
                invalidation_reasons=reasons,
            )
        )
    return boxes, malformed


def _group_boxes(boxes: list[BoxEvent], region: str, offset: int) -> list[BoxGroup]:
    valid = [box for box in boxes if box.structurally_valid and box.candidate is not None]
    raw_groups: list[list[BoxEvent]] = []
    current: list[BoxEvent] = []
    for box in valid:
        if current:
            gap = region[current[-1].span[1] - offset : box.span[0] - offset]
            display_boundary = bool(re.search(r"\\\]\s*\\\[|\$\$\s*\$\$", gap))
            adjacent = len(gap) <= 80 and bool(ADJACENT_BOX_GAP_RE.fullmatch(gap))
            if display_boundary or not adjacent:
                raw_groups.append(current)
                current = []
        current.append(box)
    if current:
        raw_groups.append(current)

    groups: list[BoxGroup] = []
    for members in raw_groups:
        components = [str(member.candidate) for member in members]
        candidate = ", ".join(components)
        start, end = members[0].span[0], members[-1].span[1]
        groups.append(
            BoxGroup(
                event_id=f"boxed_group_{len(groups):04d}",
                span=[start, end],
                raw=region[start - offset : end - offset],
                candidate=candidate,
                components=components,
                source_event_ids=[member.event_id for member in members],
                candidate_sha256=_sha256(candidate),
            )
        )
    return groups


def extract_last_boxed_group(
    text: str,
) -> tuple[BoxGroup | None, list[BoxEvent], list[BoxGroup], list[dict[str, Any]], dict[str, Any]]:
    region, offset, region_name, think = answer_region(text)
    boxes, malformed = _balanced_boxes(region, offset)
    groups = _group_boxes(boxes, region, offset)
    selected = groups[-1] if groups else None
    region_info = {
        "name": region_name,
        "span": [offset, offset + len(region)],
        "sha256": _sha256(region),
        "think_tags": think,
    }
    return selected, boxes, groups, malformed, region_info


def verify_equivalence(
    candidate: str,
    gold: str,
    timeout_seconds: int = 15,
    *,
    verifier: Callable[[str, str], dict[str, Any]] | None = None,
    force_scalar: bool = False,
) -> dict[str, Any]:
    return v2.verify_equivalence(
        candidate,
        gold,
        timeout_seconds=timeout_seconds,
        verifier=verifier,
        force_scalar=force_scalar,
    )


def _empty_verification(gold_answer: str) -> dict[str, Any]:
    return {
        "parse_success": False,
        "gold_parsed": None,
        "candidate_parsed": None,
        "equivalent": False,
        "exception": None,
        "timeout": False,
        "verification_type": None,
        "candidate_components": [],
        "gold_components": v2._split_top_level(gold_answer),
        "cardinality_match": False,
    }


def _counterfactual_box_score(
    selected: BoxGroup | None,
    boxes: list[BoxEvent],
    malformed_boxes: list[dict[str, Any]],
    gold_answer: str,
    *,
    parse_fn: Callable[..., Any] | None,
    verifier: Callable[[str, str], dict[str, Any]] | None,
) -> dict[str, Any]:
    verification = _empty_verification(gold_answer)
    parseable = False
    parse_detail: str | None = None
    decision_reason = "unscorable_no_box"
    if selected is None:
        if malformed_boxes or any(not box.structurally_valid for box in boxes):
            decision_reason = "unscorable_malformed_or_empty_box"
    else:
        parseable, parse_detail = v1.parse_candidate(selected.candidate, parse_fn=parse_fn)
        if not parseable:
            decision_reason = "unscorable_parse_failure"
            verification["exception"] = parse_detail
        else:
            verification = verify_equivalence(selected.candidate, gold_answer, verifier=verifier)
            if not verification.get("parse_success"):
                decision_reason = "unscorable_verifier_failure"
            elif verification.get("equivalent"):
                decision_reason = "correct"
            else:
                decision_reason = "incorrect_math"
    is_scorable = bool(selected is not None and parseable and verification.get("parse_success"))
    return {
        "candidate": selected.candidate if selected else None,
        "candidate_sha256": _sha256(selected.candidate if selected else None),
        "selected_box_group_id": selected.event_id if selected else None,
        "parseable": parseable,
        "parse_result": parse_detail,
        "is_scorable": is_scorable,
        "is_correct": bool(is_scorable and verification.get("equivalent")),
        "decision_reason": decision_reason,
        "verification": verification,
    }


def score_response(
    generated_text: str,
    gold_answer: str,
    *,
    finish_reason: str | None,
    stop_token_id: int | None,
    cap_hit: bool,
    problem_id: str | int,
    parse_fn: Callable[..., Any] | None = None,
    verifier: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = clean_text(generated_text)
    selected, boxes, groups, malformed_boxes, region = extract_last_boxed_group(text)
    counterfactual = _counterfactual_box_score(
        selected,
        boxes,
        malformed_boxes,
        gold_answer,
        parse_fn=parse_fn,
        verifier=verifier,
    )
    if cap_hit:
        official_candidate = None
        official_is_scorable = False
        official_is_correct = False
        official_reason = "cap_hit"
    else:
        official_candidate = counterfactual["candidate"]
        official_is_scorable = bool(counterfactual["is_scorable"])
        official_is_correct = bool(counterfactual["is_correct"])
        official_reason = str(counterfactual["decision_reason"])

    selected_history = (
        [
            {
                "action": "selected_chronologically_last_complete_boxed_group",
                "event_id": selected.event_id,
                "ignored_later_reasoning": True,
            }
        ]
        if selected
        else [{"action": "no_complete_boxed_group"}]
    )
    verification = counterfactual["verification"]
    return {
        "scorer_version": SCORER_VERSION,
        "trace_schema_version": "math500_strict_boxed_trace_v3",
        "problem_id": str(problem_id),
        "official_extracted_candidate": official_candidate,
        "official_candidate_sha256": _sha256(official_candidate),
        "is_scorable": official_is_scorable,
        "official_decision_reason": official_reason,
        "is_correct": official_is_correct,
        "extraction_region": region["name"],
        "extraction_region_span": region["span"],
        "extraction_region_sha256": region["sha256"],
        "extraction_rule": (
            "last_adjacent_boxed_group"
            if selected and len(selected.components) > 1
            else "last_complete_box"
            if selected
            else "no_complete_box"
        ),
        "think_tag_state": region["think_tags"]["state"],
        "think_tag_counts": {
            "open": region["think_tags"]["open_count"],
            "close": region["think_tags"]["close_count"],
        },
        "think_tag_spans": {
            "open": region["think_tags"]["open_spans"],
            "close": region["think_tags"]["close_spans"],
        },
        "think_closure": region["think_tags"]["state"] == "complete",
        "box_events": [asdict(box) for box in boxes],
        "boxed_groups": [asdict(group) for group in groups],
        "selected_box_group": asdict(selected) if selected else None,
        "selected_event_ids": [selected.event_id] if selected else [],
        "selection_history": selected_history,
        "candidate_event_history": [asdict(group) for group in groups],
        "invalidation_flags": {"malformed_boxes": malformed_boxes},
        "boxed_answer_candidate": selected.candidate if selected else None,
        "last_confident_answer_candidate": selected.candidate if selected else None,
        "answer_anywhere_diagnostic": selected.candidate if selected else None,
        "cap_counterfactual_diagnostic": counterfactual if cap_hit else None,
        "typed_verifier_evidence": verification,
        "parse_result": counterfactual["parse_result"],
        "math_verify_result": official_is_correct,
        "finish_reason": finish_reason,
        "stop_token_observed": stop_token_id,
        "cap_status": bool(cap_hit),
        "answer_replacement_count": max(0, len(groups) - 1),
        "partial_cap_suppression_count": 0,
        "structural_candidate_rejection_count": sum(not box.structurally_valid for box in boxes),
        "parser_verifier_exception": verification.get("exception"),
        "parser_verifier_timeout": bool(verification.get("timeout")),
        "generated_text_sha256": _sha256(generated_text),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
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
        )
    )


if __name__ == "__main__":
    main()
