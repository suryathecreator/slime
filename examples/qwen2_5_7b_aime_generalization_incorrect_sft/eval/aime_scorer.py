#!/usr/bin/env python3
"""Whole-response, strict last-box scorer for AIME integer answers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

SCORER_VERSION = "aime_last_boxed_integer_scorer_v1"
TRACE_SCHEMA_VERSION = "aime_last_boxed_integer_trace_v1"
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
BOX_START_RE = re.compile(r"\\boxed\s*\{")
INTEGER_RE = re.compile(r"\+?\d+")
WRAPPER_RE = re.compile(r"(?s)^\\(?:text|mathrm|mathbf|mathit|mbox)\s*\{(?P<value>.*)\}$")


@dataclass(frozen=True)
class BoxEvent:
    event_id: str
    span: tuple[int, int]
    raw: str
    interior_raw: str | None
    complete: bool
    valid: bool
    invalidation_reasons: tuple[str, ...] = ()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean_generated_text(text: str | None) -> str:
    result = str(text or "")
    for token in SPECIAL_TOKENS:
        result = result.replace(token, "")
    return result


def is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    cursor = position - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return bool(backslashes % 2)


def scan_box_events(text: str) -> tuple[list[BoxEvent], list[dict[str, Any]]]:
    events: list[BoxEvent] = []
    malformed: list[dict[str, Any]] = []
    for match in BOX_START_RE.finditer(text):
        cursor = match.end()
        depth = 1
        while cursor < len(text) and depth:
            if text[cursor] == "{" and not is_escaped(text, cursor):
                depth += 1
            elif text[cursor] == "}" and not is_escaped(text, cursor):
                depth -= 1
            cursor += 1
        if depth:
            malformed.append(
                {
                    "raw": text[match.start() :],
                    "reason": "unclosed_box",
                    "span": [match.start(), len(text)],
                }
            )
            continue
        interior = text[match.end() : cursor - 1]
        reasons = ("empty_box",) if not interior.strip() else ()
        events.append(
            BoxEvent(
                event_id=f"box_{len(events):04d}",
                span=(match.start(), cursor),
                raw=text[match.start() : cursor],
                interior_raw=interior,
                complete=True,
                valid=not reasons,
                invalidation_reasons=reasons,
            )
        )
    return events, malformed


def _strip_math_delimiter(value: str) -> str:
    if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
        return value[1:-1].strip()
    if len(value) >= 4 and value.startswith(r"\(") and value.endswith(r"\)"):
        return value[2:-2].strip()
    if len(value) >= 4 and value.startswith(r"\[") and value.endswith(r"\]"):
        return value[2:-2].strip()
    return value


def _unwrap_fixed_presentation(value: str) -> str:
    result = value.strip()
    for _ in range(8):
        stripped = _strip_math_delimiter(result)
        wrapper = WRAPPER_RE.fullmatch(stripped)
        updated = wrapper.group("value").strip() if wrapper else stripped
        if updated == result:
            return result
        result = updated
    return result


def parse_aime_integer(value: str | None) -> int | None:
    if value is None:
        return None
    candidate = _unwrap_fixed_presentation(value)
    if len(candidate) > 32 or INTEGER_RE.fullmatch(candidate) is None:
        return None
    parsed = int(candidate, 10)
    return parsed if 0 <= parsed <= 999 else None


def score_response(
    generated_text: str,
    gold_answer: int | str,
    *,
    finish_reason: str | None,
    stop_token_id: int | None,
    cap_hit: bool,
    problem_id: str | int,
) -> dict[str, Any]:
    gold = parse_aime_integer(str(gold_answer))
    if gold is None:
        raise ValueError(f"invalid AIME gold for {problem_id}: {gold_answer!r}")

    text = clean_generated_text(generated_text)
    events, malformed = scan_box_events(text)
    valid = [event for event in events if event.valid]
    selected = valid[-1] if valid else None
    candidate_raw = selected.interior_raw if selected is not None else None
    parsed = parse_aime_integer(candidate_raw)
    if selected is None:
        reason = "no_valid_complete_box"
    elif parsed is None:
        reason = "integer_parse_failure"
    elif parsed == gold:
        reason = "correct"
    else:
        reason = "integer_mismatch"
    correct = reason == "correct"
    return {
        "box_events": [asdict(event) for event in events],
        "cap_status": bool(cap_hit),
        "finish_reason": finish_reason,
        "generated_text_sha256": sha256_text(generated_text),
        "gold_integer": gold,
        "is_correct": correct,
        "is_scorable": parsed is not None,
        "malformed_boxes": malformed,
        "official_candidate_raw": candidate_raw,
        "official_candidate_sha256": sha256_text(candidate_raw or ""),
        "official_decision_reason": reason,
        "official_extracted_source_raw": selected.raw if selected else None,
        "official_extracted_source_span": list(selected.span) if selected else None,
        "parsed_integer": parsed,
        "problem_id": str(problem_id),
        "scorer_version": SCORER_VERSION,
        "selected_box_event_id": selected.event_id if selected else None,
        "stop_token_observed": stop_token_id,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
    }
