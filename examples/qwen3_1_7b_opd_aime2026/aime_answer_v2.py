#!/usr/bin/env python3
"""Auditable AIME integer extraction and exact-arithmetic scoring, version 2."""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from fractions import Fraction
from typing import Any


SCORER_VERSION = "aime_answer_v2"
_THINK_CLOSE = "</think>"
_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
_MARKER_RE = re.compile(
    r"(?im)(?:final\s+(?:answer|result)\s*(?:is|:|=)?|"
    r"(?:the\s+)?answer\s+(?:is|equals)|answer\s*:|"
    r"my\s+(?:final\s+)?answer\s*(?:is|:|=)?)\s*"
)
_RETRACTION_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:that|this|the|my|our)\s+(?:(?:last|previous|final)\s+)?(?:answer|result)\s+is\s+(?:wrong|incorrect|false)|"
    r"i\s+(?:was|am)\s+wrong|"
    r"i\s+(?:retract|reject|discard)\s+(?:that|this|the\s+answer|my\s+answer)|"
    r"(?:reject|discard)\s+(?:that|this|the\s+answer|the\s+result)"
    r")\b"
)
_INTEGER_RE = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})*(?!\w|\.\d|\s*/)")
_ANSWER_LEAD_RE = re.compile(r"(?i)\b(?:therefore|thus|hence|so|consequently|answer)\b")
_LATEX_FORMAT_RE = re.compile(r"\\(?:text|mathrm|mathbf|mathit)\s*\{([^{}]*)\}")
_MATH_PREFIX_RE = re.compile(r"^(?:\s|\d|[,+\-*/=().{}]|\\(?:cdot|times)\b)+")
_TRAILING_EXPLANATION_RE = re.compile(
    r"(?is)^(.+?)(?:,?\s+(?:because|based\s+on|according\s+to|which\s+(?:means|shows)|"
    r"as\s+(?:computed|calculated)|from\s+the\s+calculation)\b).*$"
)


SCORER_FIELDS = frozenset(
    {
        "scorer_version",
        "grade",
        "is_correct",
        "candidate",
        "parsed_answer",
        "gold_integer",
        "extraction_method",
        "decision_reason",
        "answer_events",
        "selected_event",
        "selected_event_id",
        "selection_trace",
        "region_policy",
        "selection_region_span",
        "generated_text_sha256",
        "replacement_count",
        "reaffirmation_count",
        "retraction_count",
        "parse_failure",
        "cap_hit",
        "finish_reason",
        "problem_id",
    }
)


def _clean_generated_text(generated_text: str | None) -> str:
    text = str(generated_text or "")
    for token in _SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def _answer_region(text: str) -> tuple[str, int, str]:
    if _THINK_CLOSE in text:
        offset = text.rfind(_THINK_CLOSE) + len(_THINK_CLOSE)
        return text[offset:], offset, "final_post_think_region"
    return text, 0, "tagless_full_output"


def _event(
    event_type: str,
    start: int,
    end: int,
    raw: str,
    candidate: str | None,
    *,
    strength: str,
    eligible: bool,
) -> dict[str, Any]:
    return {
        "event_id": "",
        "event_type": event_type,
        "strength": strength,
        "eligible": bool(eligible),
        "candidate": candidate,
        "raw": raw,
        "span": [start, end],
    }


def _balanced_boxes(text: str, offset: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            slash_count = 0
            slash_cursor = cursor - 1
            while slash_cursor >= 0 and text[slash_cursor] == "\\":
                slash_count += 1
                slash_cursor -= 1
            escaped = slash_count % 2 == 1
            if not escaped and text[cursor] == "{":
                depth += 1
            elif not escaped and text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            candidate = text[match.end() : cursor - 1].strip()
            events.append(
                _event(
                    "balanced_box",
                    offset + match.start(),
                    offset + cursor,
                    text[match.start() : cursor],
                    candidate or None,
                    strength="strong",
                    eligible=bool(candidate),
                )
            )
    return events


def _marker_candidate(tail: str, offset: int) -> tuple[str, int] | None:
    leading = len(tail) - len(tail.lstrip())
    stripped = tail.lstrip()
    boxes = _balanced_boxes(stripped, offset + leading)
    if boxes and boxes[0]["span"][0] == offset + leading:
        box = boxes[0]
        return str(box["candidate"]), int(box["span"][1]) - offset
    explanation = _TRAILING_EXPLANATION_RE.match(stripped)
    if explanation:
        stripped = explanation.group(1).rstrip()
    prefix = _MATH_PREFIX_RE.match(stripped)
    if prefix is None or not re.search(r"\d", prefix.group(0)):
        return None
    candidate = prefix.group(0).rstrip()
    candidate = candidate.rstrip(" ,:;!`")
    if not candidate:
        return None
    return candidate, leading + len(prefix.group(0).rstrip())


def _explicit_events(text: str, offset: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for marker in _MARKER_RE.finditer(text):
        line_end = text.find("\n", marker.end())
        if line_end < 0:
            line_end = len(text)
        found = _marker_candidate(text[marker.end() : line_end], offset + marker.end())
        if found is None:
            continue
        candidate, local_end = found
        end = marker.end() + local_end
        events.append(
            _event(
                "explicit_answer_marker",
                offset + marker.start(),
                offset + end,
                text[marker.start() : end],
                candidate,
                strength="strong",
                eligible=True,
            )
        )
    return events


def _line_is_answer_like(line: str, number: re.Match[str]) -> tuple[bool, str]:
    stripped = line.strip()
    without_wrappers = stripped
    for wrapper in ("$", "\\(", "\\)", "\\[", "\\]"):
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
    eligible: list[dict[str, Any]] = []
    for line in text.splitlines(keepends=True):
        matches = list(_INTEGER_RE.finditer(line))
        if matches:
            number = matches[-1]
            accepted, heuristic = _line_is_answer_like(line, number)
            if accepted:
                event = _event(
                    "unboxed_final_numeric",
                    offset + line_offset + number.start(),
                    offset + line_offset + number.end(),
                    number.group(0),
                    number.group(0),
                    strength="weak",
                    eligible=True,
                )
                event["heuristic"] = heuristic
                eligible.append(event)
        line_offset += len(line)
    if not eligible:
        return None
    last = eligible[-1]
    trailing = text[int(last["span"][1]) - offset :].strip()
    if trailing and not re.fullmatch(r"[\s\W_]*", trailing):
        return None
    return last


class _ExactArithmetic(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes = 0

    def visit(self, node: ast.AST) -> Fraction:  # type: ignore[override]
        self.nodes += 1
        if self.nodes > 64:
            raise ValueError("arithmetic expression is too complex")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> Fraction:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("only integer literals are supported")
        if abs(node.value) > 10**12:
            raise ValueError("integer literal is too large")
        return Fraction(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Fraction:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Fraction:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        raise ValueError("unsupported binary operator")

    def generic_visit(self, node: ast.AST) -> Fraction:
        raise ValueError(f"unsupported arithmetic syntax: {type(node).__name__}")


def _normalize_arithmetic(candidate: str) -> str:
    value = str(candidate).strip()
    for _ in range(4):
        updated = _LATEX_FORMAT_RE.sub(r"\1", value)
        if updated == value:
            break
        value = updated
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
    if value.startswith(r"\(") and value.endswith(r"\)"):
        value = value[2:-2].strip()
    if value.startswith(r"\[") and value.endswith(r"\]"):
        value = value[2:-2].strip()
    value = value.replace(r"\,", "").replace(",", "")
    value = re.sub(r"\\(?:cdot|times)\b", "*", value)
    value = value.replace("{", "(").replace("}", ")")
    return value.rstrip().rstrip(".,:;!`").strip()


def _evaluate_expression(expression: str) -> Fraction:
    if not expression or len(expression) > 256:
        raise ValueError("empty or overlong arithmetic expression")
    if not re.fullmatch(r"[\d\s+\-*/()]+", expression):
        raise ValueError("expression contains unsupported characters")
    tree = ast.parse(expression, mode="eval")
    return _ExactArithmetic().visit(tree)


def _canonical_integer(candidate: str | None) -> int | None:
    if candidate is None:
        return None
    try:
        value = _normalize_arithmetic(candidate)
        sides = value.split("=")
        if any(not side.strip() for side in sides):
            return None
        results = [_evaluate_expression(side.strip()) for side in sides]
        if len(results) > 1 and any(result != results[0] for result in results[1:]):
            return None
        result = results[-1]
        if result.denominator != 1 or not 0 <= result.numerator <= 999:
            return None
        return int(result.numerator)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def _extract_events(region: str, offset: int) -> list[dict[str, Any]]:
    events = _balanced_boxes(region, offset) + _explicit_events(region, offset)
    weak = _weak_final_event(region, offset)
    if weak is not None:
        events.append(weak)
    for match in _RETRACTION_RE.finditer(region):
        events.append(
            _event(
                "explicit_retraction",
                offset + match.start(),
                offset + match.end(),
                match.group(0),
                None,
                strength="control",
                eligible=False,
            )
        )
    events.sort(key=lambda event: (int(event["span"][0]), int(event["span"][1]), str(event["event_type"])))
    for index, answer_event in enumerate(events):
        answer_event["event_id"] = f"event_{index:04d}"
        if answer_event.get("eligible"):
            answer_event["parsed_candidate"] = _canonical_integer(answer_event.get("candidate"))
    return events


def _select_answer(events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool]:
    selected: dict[str, Any] | None = None
    retracted = False
    trace: list[dict[str, Any]] = []
    for answer_event in events:
        if answer_event["event_type"] == "explicit_retraction":
            if selected is None:
                trace.append({"action": "ignored_retraction_without_answer", "event_id": answer_event["event_id"]})
            else:
                retracted = True
                trace.append(
                    {
                        "action": "retraction",
                        "event_id": answer_event["event_id"],
                        "retracted_event_id": selected["event_id"],
                    }
                )
            continue
        if not answer_event.get("eligible") or not answer_event.get("candidate"):
            continue
        if selected is None:
            selected = answer_event
            retracted = False
            trace.append({"action": "initial_selection", "event_id": answer_event["event_id"]})
            continue
        previous = selected
        selected = answer_event
        if retracted:
            action = "replacement_after_retraction"
        elif previous.get("parsed_candidate") is not None and previous.get("parsed_candidate") == answer_event.get("parsed_candidate"):
            action = "reaffirmation"
        else:
            action = "replacement"
        trace.append({"action": action, "event_id": answer_event["event_id"], "replaced_event_id": previous["event_id"]})
        retracted = False
    return selected, trace, selected is not None and retracted


def score_answer(
    generated_text: str,
    gold_answer: str | int,
    problem_text: str,
    problem_id: str,
    cap_hit: bool,
    finish_reason: str | None,
) -> dict[str, Any]:
    del problem_text
    text = _clean_generated_text(generated_text)
    region, offset, region_policy = _answer_region(text)
    events = _extract_events(region, offset)
    selected, trace, unresolved_retraction = _select_answer(events)
    candidate = str(selected["candidate"]) if selected is not None else None
    parsed = _canonical_integer(candidate)
    gold = _canonical_integer(str(gold_answer))
    if gold is None:
        raise ValueError(f"AIME gold answer is not a valid integer for {problem_id}: {gold_answer!r}")

    if selected is None:
        is_correct = False
        reason = "no_eligible_answer_event"
    elif unresolved_retraction:
        is_correct = False
        reason = "explicit_retraction_without_replacement"
    elif parsed is None:
        is_correct = False
        reason = "arithmetic_parse_failure"
    elif parsed == gold:
        is_correct = True
        reason = "exact_integer_match"
    else:
        is_correct = False
        reason = "integer_mismatch"

    return {
        "scorer_version": SCORER_VERSION,
        "grade": "correct" if is_correct else "incorrect",
        "is_correct": is_correct,
        "candidate": candidate,
        "parsed_answer": parsed,
        "gold_integer": gold,
        "extraction_method": selected["event_type"] if selected is not None else "none",
        "decision_reason": reason,
        "answer_events": events,
        "selected_event": selected,
        "selected_event_id": selected["event_id"] if selected is not None else None,
        "selection_trace": trace,
        "region_policy": region_policy,
        "selection_region_span": [offset, len(text)],
        "generated_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "replacement_count": sum(item["action"].startswith("replacement") for item in trace),
        "reaffirmation_count": sum(item["action"] == "reaffirmation" for item in trace),
        "retraction_count": sum(item["action"] == "retraction" for item in trace),
        "parse_failure": selected is None or (parsed is None and not unresolved_retraction),
        "cap_hit": bool(cap_hit),
        "finish_reason": finish_reason,
        "problem_id": str(problem_id),
    }


def metrics_from_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot score an empty prediction set")
    correct = sum(bool(row.get("is_correct")) for row in predictions)
    cap_rows = [row for row in predictions if row.get("cap_hit")]
    trace_counts = Counter(
        str(item.get("action"))
        for row in predictions
        for item in row.get("selection_trace", [])
    )
    return {
        "num_generations": len(predictions),
        "total_correct": correct,
        "pass_at_1_mc": correct / len(predictions),
        "cap_hit_count": len(cap_rows),
        "cap_hit_rate": len(cap_rows) / len(predictions),
        "cap_hit_accuracy": (sum(bool(row.get("is_correct")) for row in cap_rows) / len(cap_rows)) if cap_rows else 0.0,
        "parse_failure_count": sum(bool(row.get("parse_failure", row.get("parsed_answer") is None)) for row in predictions),
        "extraction_method_counts": dict(sorted(Counter(str(row.get("extraction_method")) for row in predictions).items())),
        "decision_reason_counts": dict(sorted(Counter(str(row.get("decision_reason")) for row in predictions).items())),
        "region_policy_counts": dict(sorted(Counter(str(row.get("region_policy")) for row in predictions).items())),
        "selection_action_counts": dict(sorted(trace_counts.items())),
        "replacement_count": sum(int(row.get("replacement_count", 0)) for row in predictions),
        "reaffirmation_count": sum(int(row.get("reaffirmation_count", 0)) for row in predictions),
        "retraction_count": sum(int(row.get("retraction_count", 0)) for row in predictions),
    }
