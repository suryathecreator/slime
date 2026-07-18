#!/usr/bin/env python3
"""Auditable numerical answer extraction for AIME 2026 generations."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


SCORER_VERSION = "aime_answer_v1"
_THINK_CLOSE = "</think>"
_MARKER_RE = re.compile(r"(?i)(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=)\s*")
_INTEGER_RE = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})*(?!\w|\.\d|\s*/)")
_ANSWER_LEAD_RE = re.compile(r"(?i)\b(?:therefore|thus|hence|so|consequently|answer)\b")
_LATEX_WRAPPERS = ("$", "\\(", "\\)", "\\[", "\\]")


def _balanced_boxes(text: str, offset: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "{" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth += 1
            elif text[cursor] == "}" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth -= 1
            cursor += 1
        if depth == 0:
            events.append(
                {
                    "event_type": "balanced_box",
                    "strength": "strong",
                    "candidate": text[match.end() : cursor - 1].strip(),
                    "span": [offset + match.start(), offset + cursor],
                }
            )
    return events


def _first_integer_after_marker(text: str, start: int) -> tuple[str, int, int] | None:
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    tail = text[start : min(line_end, start + 128)]
    box_events = _balanced_boxes(tail, start)
    if box_events:
        event = box_events[0]
        return str(event["candidate"]), int(event["span"][0]), int(event["span"][1])
    match = _INTEGER_RE.search(tail)
    if match is None:
        return None
    if re.search(r"[A-Za-z]", tail[: match.start()]):
        return None
    return match.group(0), start + match.start(), start + match.end()


def _explicit_events(text: str, offset: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for marker in _MARKER_RE.finditer(text):
        found = _first_integer_after_marker(text, marker.end())
        if found is None:
            continue
        candidate, _, end = found
        events.append(
            {
                "event_type": "explicit_answer_marker",
                "strength": "strong",
                "candidate": candidate,
                "span": [offset + marker.start(), offset + end],
            }
        )
    return events


def _line_is_answer_like(line: str, number: re.Match[str]) -> tuple[bool, str]:
    stripped = line.strip()
    without_wrappers = stripped
    for wrapper in _LATEX_WRAPPERS:
        without_wrappers = without_wrappers.replace(wrapper, "")
    bare = without_wrappers.strip(" .,:;!`*_{}")
    candidate = number.group(0).replace(",", "")
    if bare in {number.group(0), candidate, f"+{candidate}"}:
        return True, "standalone_final_numeric"
    if _ANSWER_LEAD_RE.search(stripped):
        suffix = stripped[number.end() :].strip(" .,:;!`*_{}$\\)")
        if not suffix or suffix.lower() in {"is the answer", "is our answer"}:
            return True, "conclusion_numeric"
    return False, "incidental_numeric"


def _weak_final_event(text: str, offset: int) -> dict[str, Any] | None:
    line_offset = 0
    candidates: list[dict[str, Any]] = []
    for line in text.splitlines(keepends=True):
        matches = list(_INTEGER_RE.finditer(line))
        if matches:
            number = matches[-1]
            accepted, reason = _line_is_answer_like(line, number)
            candidates.append(
                {
                    "event_type": "unboxed_final_numeric",
                    "strength": "weak" if accepted else "audit_only",
                    "candidate": number.group(0),
                    "span": [offset + line_offset + number.start(), offset + line_offset + number.end()],
                    "heuristic": reason,
                    "eligible": accepted,
                }
            )
        line_offset += len(line)
    eligible = [event for event in candidates if event["eligible"]]
    if not eligible:
        return None
    last = eligible[-1]
    trailing = text[last["span"][1] - offset :].strip()
    if trailing and not re.fullmatch(r"[\s\W_]*", trailing):
        return None
    return last


def _canonical_integer(candidate: str | None) -> int | None:
    if candidate is None:
        return None
    value = str(candidate).strip()
    value = re.sub(r"\\(?:text|mathrm|mathbf|mathit)\s*\{([^{}]*)\}", r"\1", value)
    for token in ("$", "\\(", "\\)", "\\[", "\\]", "{", "}", "\\,"):
        value = value.replace(token, "")
    value = value.replace(",", "").strip()
    if not re.fullmatch(r"[-+]?\d+", value):
        return None
    return int(value)


def score_answer(
    generated_text: str,
    gold_answer: str | int,
    problem_text: str,
    problem_id: str,
    cap_hit: bool,
    finish_reason: str | None,
) -> dict[str, Any]:
    del problem_text
    text = str(generated_text)
    if cap_hit:
        region, offset, region_policy = text, 0, "unfinished_full_trace"
    elif _THINK_CLOSE in text:
        offset = text.rfind(_THINK_CLOSE) + len(_THINK_CLOSE)
        region, region_policy = text[offset:], "final_post_think_region"
    else:
        region, offset, region_policy = text, 0, "tagless_full_output"

    events = _balanced_boxes(region, offset) + _explicit_events(region, offset)
    events.sort(key=lambda event: (int(event["span"][0]), int(event["span"][1])))
    selected = events[-1] if events else _weak_final_event(region, offset)
    if selected is not None and selected not in events:
        events.append(selected)
    candidate = str(selected["candidate"]) if selected is not None else None
    parsed = _canonical_integer(candidate)
    gold = _canonical_integer(str(gold_answer))
    if gold is None:
        raise ValueError(f"AIME gold answer is not an integer for {problem_id}: {gold_answer!r}")
    is_correct = parsed is not None and parsed == gold
    return {
        "scorer_version": SCORER_VERSION,
        "grade": "correct" if is_correct else "incorrect",
        "is_correct": is_correct,
        "candidate": candidate,
        "parsed_answer": parsed,
        "gold_integer": gold,
        "extraction_method": selected["event_type"] if selected is not None else "none",
        "decision_reason": "exact_integer_match" if is_correct else "no_eligible_answer_event" if selected is None else "integer_parse_failure" if parsed is None else "integer_mismatch",
        "answer_events": events,
        "selected_event": selected,
        "region_policy": region_policy,
        "cap_hit": bool(cap_hit),
        "finish_reason": finish_reason,
        "problem_id": str(problem_id),
    }


def metrics_from_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot score an empty prediction set")
    correct = sum(bool(row.get("is_correct")) for row in predictions)
    cap_rows = [row for row in predictions if row.get("cap_hit")]
    return {
        "num_generations": len(predictions),
        "total_correct": correct,
        "pass_at_1_mc": correct / len(predictions),
        "cap_hit_count": len(cap_rows),
        "cap_hit_rate": len(cap_rows) / len(predictions),
        "cap_hit_accuracy": (sum(bool(row.get("is_correct")) for row in cap_rows) / len(cap_rows)) if cap_rows else 0.0,
        "parse_failure_count": sum(row.get("parsed_answer") is None for row in predictions),
        "extraction_method_counts": dict(sorted(Counter(str(row.get("extraction_method")) for row in predictions).items())),
        "decision_reason_counts": dict(sorted(Counter(str(row.get("decision_reason")) for row in predictions).items())),
    }
