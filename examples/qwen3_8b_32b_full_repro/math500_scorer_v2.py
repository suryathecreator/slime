#!/usr/bin/env python3
"""Versioned MATH-500 answer-event scorer with cap-safe selection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Callable

from examples.qwen3_8b_32b_full_repro import math500_scorer_v1 as v1


SCORER_VERSION = "math500_event_scorer_v2"
THINK_CLOSE = "</think>"
SPECIAL_TOKENS = v1.SPECIAL_TOKENS
ASSERTION_RE = v1.ASSERTION_RE
CONCLUSION_RE = v1.CONCLUSION_RE
EXPLICIT_RETRACTION_RE = re.compile(
    r"(?i)\b(?:i\s+(?:retract|reject|discard)\s+(?:that|this|my|the)?\s*(?:answer|result)?|"
    r"(?:that|this|my|the)\s+(?:answer|result)\s+is\s+(?:wrong|incorrect|false)|"
    r"scratch\s+that|not\s+(?:the\s+)?(?:answer|result))\b"
)
STRUCTURAL_ONLY_RE = re.compile(
    r"^(?:\s|\$|\\\[|\\\]|\\\(|\\\)|\\begin\{(?:displaymath|equation\*?)\}|"
    r"\\end\{(?:displaymath|equation\*?)\})+$"
)
MATHISH_RE = v1.MATHISH_RE
DOUBT_RE = v1.DOUBT_RE


@dataclass
class AnswerEvent:
    event_id: str
    event_type: str
    span: list[int]
    raw: str
    candidate: str | None
    confidence: str
    components: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    parseable: bool = False
    parse_result: str | None = None
    parser_exception: str | None = None
    parser_timeout: bool = False
    complete: bool = True
    terminal_at_cap: bool = False
    invalidated: bool = False
    retracted: bool = False
    doubtful: bool = False
    superseded: bool = False
    invalidation_reasons: list[str] = field(default_factory=list)
    candidate_sha256: str | None = None


def clean_text(text: str | None) -> str:
    return v1.clean_text(text)


def answer_region(text: str) -> tuple[str, int, str, bool]:
    if THINK_CLOSE in text:
        offset = text.rfind(THINK_CLOSE) + len(THINK_CLOSE)
        return text[offset:], offset, "final_post_think_region", True
    return text, 0, "unfinished_full_response", False


def is_structural_only(candidate: str | None) -> bool:
    value = str(candidate or "").strip()
    return not value or bool(STRUCTURAL_ONLY_RE.fullmatch(value))


def _balanced_boxes(text: str, offset: int) -> tuple[list[AnswerEvent], list[dict[str, Any]]]:
    events: list[AnswerEvent] = []
    malformed: list[dict[str, Any]] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        cursor = match.end()
        depth = 1
        while cursor < len(text) and depth:
            escaped = cursor > 0 and text[cursor - 1] == "\\"
            if text[cursor] == "{" and not escaped:
                depth += 1
            elif text[cursor] == "}" and not escaped:
                depth -= 1
            cursor += 1
        if depth:
            malformed.append({"span": [offset + match.start(), offset + len(text)], "raw": text[match.start() :]})
            continue
        candidate = text[match.end() : cursor - 1].strip()
        events.append(
            AnswerEvent(
                event_id="",
                event_type="boxed",
                span=[offset + match.start(), offset + cursor],
                raw=text[match.start() : cursor],
                candidate=candidate or None,
                confidence="confident" if candidate and not is_structural_only(candidate) else "invalid",
                components=[candidate] if candidate else [],
            )
        )
    return events, malformed


def _trim_candidate(raw: str) -> str | None:
    value = raw.strip()
    if not value:
        return None
    boxes, _ = _balanced_boxes(value, 0)
    if boxes and boxes[0].span[0] == 0:
        return boxes[0].candidate
    value = re.split(r"</think>|<\|(?:im_end|endoftext)\|>", value, maxsplit=1)[0]
    value = re.split(r"(?i)\b(?:wait|actually|alternatively|scratch\s+that|however)\b", value, maxsplit=1)[0]
    value = re.split(
        r"(?i),\s*(?:but|however|so|because|since|which\s+means|as\s+required|and\s+(?:the|this|that|i|we|it)\b)",
        value,
        maxsplit=1,
    )[0]
    value = re.split(r"\.(?=\s|$)|;", value, maxsplit=1)[0]
    value = value.strip().lstrip(":=").strip()
    value = re.sub(r"(?i)^(?:indeed|probably|approximately|about)\s+", "", value)
    value = value.strip(" `*_.,;:!?$")
    if not value or is_structural_only(value) or not MATHISH_RE.search(value):
        return None
    return value


def _marked_events(
    text: str,
    offset: int,
    pattern: re.Pattern[str],
    event_type: str,
) -> tuple[list[AnswerEvent], int]:
    events: list[AnswerEvent] = []
    structural_rejections = 0
    for marker in pattern.finditer(text):
        line_end = text.find("\n", marker.end())
        if line_end < 0:
            line_end = len(text)
        raw_tail = text[marker.end() : line_end]
        candidate = _trim_candidate(raw_tail)
        if candidate is None:
            structural_rejections += int(is_structural_only(raw_tail.strip(" `*_.,;:!?$")))
            continue
        relative_end = raw_tail.find(candidate)
        event_end = marker.end() + relative_end + len(candidate) if relative_end >= 0 else line_end
        local = text[max(0, marker.start() - 32) : min(len(text), event_end + 32)]
        events.append(
            AnswerEvent(
                event_id="",
                event_type=event_type,
                span=[offset + marker.start(), offset + event_end],
                raw=text[marker.start() : line_end],
                candidate=candidate,
                confidence="confident",
                components=[candidate],
                doubtful=bool(DOUBT_RE.search(local)),
            )
        )
    return events, structural_rejections


def _standalone_final_event(text: str, offset: int) -> AnswerEvent | None:
    nonempty = [line for line in re.finditer(r"(?m)^.*$", text) if line.group(0).strip()]
    if not nonempty:
        return None
    line = nonempty[-1]
    raw = line.group(0).strip()
    if len(raw) > 512 or is_structural_only(raw) or DOUBT_RE.search(raw):
        return None
    candidate = raw.strip(" `*_")
    if not candidate or is_structural_only(candidate) or not MATHISH_RE.search(candidate):
        return None
    return AnswerEvent(
        event_id="",
        event_type="standalone_final_math_line",
        span=[offset + line.start(), offset + line.end()],
        raw=line.group(0),
        candidate=candidate,
        confidence="weak",
        components=[candidate],
    )


@lru_cache(maxsize=16384)
def _parse_cached(candidate: str) -> tuple[bool, str | None]:
    return v1.parse_candidate(candidate)


def _annotate_parse(event: AnswerEvent, parse_fn: Callable[..., Any] | None) -> None:
    if event.candidate is None or is_structural_only(event.candidate):
        event.parseable = False
        event.parser_exception = "structural_or_empty_candidate"
    else:
        result = v1.parse_candidate(event.candidate, parse_fn=parse_fn) if parse_fn else _parse_cached(event.candidate)
        event.parseable, detail = result
        if event.parseable:
            event.parse_result = detail
        else:
            event.parser_exception = detail
            event.parser_timeout = bool(detail and "timeout" in detail.lower())
    event.candidate_sha256 = hashlib.sha256(str(event.candidate or "").encode()).hexdigest()


def _box_groups(boxes: list[AnswerEvent], text: str, offset: int) -> list[AnswerEvent]:
    groups: list[list[AnswerEvent]] = []
    current: list[AnswerEvent] = []
    for box in boxes:
        if current:
            gap = text[current[-1].span[1] - offset : box.span[0] - offset]
            display_boundary = bool(re.search(r"\\\]\s*\\\[|\$\$\s*\$\$", gap))
            adjacent = len(gap) <= 80 and bool(re.fullmatch(r"[\s,;:&/]*(?:(?:and|or)\s*)?", gap, re.I))
            if display_boundary or not adjacent:
                groups.append(current)
                current = []
        current.append(box)
    if current:
        groups.append(current)
    answer_groups: list[AnswerEvent] = []
    for group in groups:
        candidates = [str(event.candidate) for event in group if event.candidate]
        candidate = ", ".join(candidates) if len(candidates) > 1 else (candidates[0] if candidates else None)
        local_start = group[0].span[0] - offset
        prefix = text[max(0, local_start - 320) : local_start]
        prefix = prefix.rsplit("\n\n", 1)[-1]
        final_marker = bool(
            re.search(r"(?is)(?:\*{0,2}final\s+answer\*{0,2}|(?:the\s+)?answer\s+is)\s*[:\n$]*$", prefix)
        )
        meta_box = not final_marker and bool(
            re.search(
                r"(?is)(?:for\s+example|e\.g\.|suppose|hypothetical(?:ly)?|"
                r"if\s+(?:the\s+)?(?:answer|roots?|solutions?|values?)\s+(?:was|were)|"
                r"(?:answer|roots?|solutions?|values?)\s+would\s+be)",
                prefix,
            )
        )
        event = AnswerEvent(
            event_id="",
            event_type="boxed_group" if len(group) > 1 else "boxed",
            span=[group[0].span[0], group[-1].span[1]],
            raw=text[group[0].span[0] - offset : group[-1].span[1] - offset],
            candidate=candidate,
            confidence="confident" if candidate and not meta_box else "invalid",
            components=candidates,
            source_event_ids=[event.event_id for event in group],
        )
        if meta_box:
            event.invalidated = True
            event.invalidation_reasons.append("meta_or_hypothetical_box")
        answer_groups.append(event)
    return answer_groups


def extract_events(
    region: str,
    offset: int,
    *,
    cap_hit: bool,
    parse_fn: Callable[..., Any] | None = None,
) -> tuple[list[AnswerEvent], list[AnswerEvent], list[dict[str, Any]], int]:
    boxes, malformed = _balanced_boxes(region, offset)
    assertions, assertion_rejections = _marked_events(region, offset, ASSERTION_RE, "explicit_assertion")
    if cap_hit:
        conclusions, conclusion_rejections = [], 0
    else:
        conclusions, conclusion_rejections = _marked_events(region, offset, CONCLUSION_RE, "conclusion")
    weak = None if cap_hit else _standalone_final_event(region, offset)
    raw_events = boxes + assertions + conclusions + ([weak] if weak and not conclusions else [])
    raw_events.sort(key=lambda event: (event.span[0], event.span[1], event.event_type))
    for index, event in enumerate(raw_events):
        event.event_id = f"answer_{index:04d}"
        _annotate_parse(event, parse_fn)
    grouped_boxes = _box_groups(boxes, region, offset)
    for index, event in enumerate(grouped_boxes):
        event.event_id = f"group_{index:04d}"
        _annotate_parse(event, parse_fn)
    strong = list(grouped_boxes)
    for assertion in assertions:
        overlaps = any(assertion.span[0] <= box.span[0] and box.span[1] <= assertion.span[1] for box in boxes)
        if not overlaps:
            strong.append(assertion)
    strong.sort(key=lambda event: (event.span[0], event.span[1], event.event_type))
    weak_events = [
        event
        for event in conclusions + ([weak] if weak and not conclusions else [])
        if event.parseable
    ]
    rejected = (
        sum(is_structural_only(event.candidate) for event in raw_events)
        + assertion_rejections
        + conclusion_rejections
    )
    return strong, weak_events, malformed, rejected


def _strip_thousands(value: str) -> str:
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r"(?<=\d),(?:\\!\s*)?(?=\d{3}(?:\D|$))", "", value)
    return value


def _split_top_level(value: str) -> list[str]:
    value = _strip_thousands(value.strip())
    if not value:
        return []
    if value.startswith("\\{") and value.endswith("\\}"):
        value = value[2:-2].strip()
    elif value.startswith("{") and value.endswith("}"):
        value = value[1:-1].strip()
    if (value.startswith("(") and value.endswith((")", "]"))) or (value.startswith("[") and value.endswith(("]", ")"))):
        return [value]
    if "\\pm" in value and value.count("\\pm") == 1:
        return [value.replace("\\pm", "+"), value.replace("\\pm", "-")]
    depth = 0
    parts: list[str] = []
    start = 0
    for index, char in enumerate(value):
        escaped = index > 0 and value[index - 1] == "\\"
        if char in "([{" and not escaped:
            depth += 1
        elif char in ")]}" and not escaped:
            depth = max(0, depth - 1)
        if depth == 0 and char == ",":
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    if len(parts) == 1:
        conjunction = re.split(r"\s+(?:and|or)\s+", value, flags=re.I)
        if len(conjunction) > 1:
            parts = [part.strip() for part in conjunction]
    return [part for part in parts if part]


def _call_verifier(candidate: str, gold: str, verifier: Callable[[str, str], dict[str, Any]] | None) -> dict[str, Any]:
    return (verifier or v1.verify_equivalence)(candidate, gold)


def verify_equivalence(
    candidate: str,
    gold: str,
    timeout_seconds: int = 15,
    *,
    verifier: Callable[[str, str], dict[str, Any]] | None = None,
    force_scalar: bool = False,
) -> dict[str, Any]:
    del timeout_seconds
    candidate_parts = [candidate] if force_scalar else _split_top_level(candidate)
    gold_parts = [gold] if force_scalar else _split_top_level(gold)
    if len(candidate_parts) == 1 and len(gold_parts) == 1:
        result = _call_verifier(candidate, gold, verifier)
        result.update(
            verification_type="scalar_or_ordered",
            candidate_components=candidate_parts,
            gold_components=gold_parts,
            cardinality_match=True,
        )
        return result
    evidence: dict[str, Any] = {
        "parse_success": True,
        "gold_parsed": repr(gold_parts),
        "candidate_parsed": repr(candidate_parts),
        "equivalent": False,
        "exception": None,
        "timeout": False,
        "verification_type": "unordered_collection_exact_cardinality",
        "candidate_components": candidate_parts,
        "gold_components": gold_parts,
        "cardinality_match": len(candidate_parts) == len(gold_parts),
        "component_matches": [],
    }
    if len(candidate_parts) != len(gold_parts):
        return evidence
    unmatched = list(range(len(gold_parts)))
    for candidate_part in candidate_parts:
        matched = None
        attempts = []
        for gold_index in unmatched:
            result = _call_verifier(candidate_part, gold_parts[gold_index], verifier)
            attempts.append({"gold_index": gold_index, "equivalent": bool(result.get("equivalent"))})
            if result.get("equivalent"):
                matched = gold_index
                break
        evidence["component_matches"].append({"candidate": candidate_part, "matched_gold_index": matched, "attempts": attempts})
        if matched is None:
            return evidence
        unmatched.remove(matched)
    evidence["equivalent"] = not unmatched
    return evidence


def _candidate_signature(event: AnswerEvent) -> list[str]:
    parts = event.components if len(event.components) > 1 else _split_top_level(str(event.candidate or ""))
    return [re.sub(r"[^a-z0-9]+", "", part.lower()) for part in parts if part]


def _terminal_at_cap(event: AnswerEvent, text: str) -> bool:
    trailing = re.sub(r"<\|(?:im_end|endoftext)\|>", "", text[event.span[1] :])
    return not trailing.strip()


def _suppress_partial_cap(events: list[AnswerEvent], selected_index: int, text: str) -> tuple[int, dict[str, Any] | None]:
    selected = events[selected_index]
    if not _terminal_at_cap(selected, text) or selected_index == 0:
        return selected_index, None
    selected_sig = _candidate_signature(selected)
    if not selected_sig:
        return selected_index, None
    previous = events[selected_index - 1]
    previous_sig = _candidate_signature(previous)
    scalar_prefix = len(selected_sig) == len(previous_sig) == 1 and len(selected_sig[0]) < len(previous_sig[0]) and previous_sig[0].startswith(selected_sig[0])
    multipart_prefix = len(selected_sig) < len(previous_sig) and previous_sig[: len(selected_sig)] == selected_sig
    if scalar_prefix or multipart_prefix:
        selected.invalidated = True
        selected.invalidation_reasons.append("terminal_strict_prefix_of_previous_complete_answer")
        return selected_index - 1, {
            "action": "partial_cap_event_suppressed",
            "suppressed_event_id": selected.event_id,
            "retained_event_id": previous.event_id,
            "rule": "strict_scalar_prefix" if scalar_prefix else "strict_multipart_prefix",
        }
    return selected_index, None


def _select_event(strong: list[AnswerEvent], weak: list[AnswerEvent], text: str, *, cap_hit: bool) -> tuple[AnswerEvent | None, list[dict[str, Any]], int]:
    history: list[dict[str, Any]] = []
    valid = [event for event in strong if event.parseable and event.confidence == "confident"]
    if valid:
        selected_index = len(valid) - 1
        partial_count = 0
        if cap_hit:
            selected_index, suppression = _suppress_partial_cap(valid, selected_index, text)
            if suppression:
                history.append(suppression)
                partial_count = 1
        selected = valid[selected_index]
        for index, event in enumerate(valid):
            if index < selected_index:
                if _candidate_signature(event) == _candidate_signature(selected):
                    history.append(
                        {
                            "action": "reaffirmation",
                            "event_id": selected.event_id,
                            "previous_event_id": event.event_id,
                        }
                    )
                else:
                    event.superseded = True
                    event.invalidated = True
                    event.invalidation_reasons.append("later_distinct_complete_answer_event")
        later_text = text[selected.span[1] :]
        retraction = EXPLICIT_RETRACTION_RE.search(later_text)
        if selected.candidate:
            tied_retraction = re.search(
                rf"(?i)\b(?:no\s*,?\s*)?not\s+{re.escape(selected.candidate.strip())}(?=\s|[.,;!?]|$)",
                later_text[:160],
            )
            if tied_retraction and (retraction is None or tied_retraction.start() < retraction.start()):
                retraction = tied_retraction
        later_valid = [event for event in valid if event.span[0] > selected.span[1]]
        if retraction and not later_valid:
            selected.retracted = True
            selected.invalidated = True
            selected.invalidation_reasons.append("explicit_unreplaced_retraction")
            history.append({"action": "explicit_unreplaced_retraction", "event_id": selected.event_id, "raw": retraction.group(0)})
            return None, history, partial_count
        history.append({"action": "selected_latest_complete_strong_event", "event_id": selected.event_id})
        return selected, history, partial_count
    if not cap_hit:
        valid_weak = [event for event in weak if event.parseable and not is_structural_only(event.candidate)]
        if valid_weak:
            selected = valid_weak[-1]
            history.append({"action": "selected_natural_stop_weak_fallback", "event_id": selected.event_id})
            return selected, history, 0
    history.append({"action": "no_complete_eligible_answer_event"})
    return None, history, 0


def _diagnostic_candidate(text: str, parse_fn: Callable[..., Any] | None) -> str | None:
    strong, weak, _malformed, _rejected = extract_events(text, 0, cap_hit=False, parse_fn=parse_fn)
    selected, _history, _partial = _select_event(strong, weak, text, cap_hit=False)
    return selected.candidate if selected else None


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
    region, offset, region_name, think_closed = answer_region(text)
    strong, weak, malformed_boxes, structural_rejections = extract_events(region, offset, cap_hit=cap_hit, parse_fn=parse_fn)
    selected, selection_history, partial_cap_suppressions = _select_event(strong, weak, text, cap_hit=cap_hit)
    candidate = selected.candidate if selected else None
    verification: dict[str, Any] = {
        "parse_success": False,
        "gold_parsed": None,
        "candidate_parsed": None,
        "equivalent": False,
        "exception": None,
        "timeout": False,
        "verification_type": None,
        "candidate_components": [],
        "gold_components": _split_top_level(gold_answer),
        "cardinality_match": False,
    }
    if candidate is not None:
        verification = verify_equivalence(
            candidate,
            gold_answer,
            verifier=verifier,
            force_scalar=bool(selected and selected.event_type == "standalone_final_math_line"),
        )
    all_events = sorted(strong + weak, key=lambda event: (event.span[0], event.span[1], event.event_type))
    extraction_exception = next(
        (event.parser_exception for event in reversed(all_events) if event.parser_exception),
        None,
    )
    boxed = next((event.candidate for event in reversed(strong) if event.event_type in {"boxed", "boxed_group"} and event.parseable), None)
    return {
        "scorer_version": SCORER_VERSION,
        "problem_id": str(problem_id),
        "official_extracted_candidate": candidate,
        "official_candidate_sha256": hashlib.sha256(str(candidate or "").encode()).hexdigest(),
        "extraction_region": region_name,
        "extraction_rule": selected.event_type if selected else "no_complete_eligible_answer_event",
        "candidate_event_history": [asdict(event) for event in all_events],
        "selection_history": selection_history,
        "selected_event_ids": [selected.event_id] if selected else [],
        "invalidation_flags": {
            "any_retraction": any(event.retracted for event in all_events),
            "any_doubt": any(event.doubtful for event in all_events),
            "any_replacement": any(event.superseded for event in all_events),
            "malformed_boxes": malformed_boxes,
        },
        "boxed_answer_candidate": boxed,
        "last_confident_answer_candidate": next((event.candidate for event in reversed(all_events) if event.parseable), None),
        "answer_anywhere_diagnostic": _diagnostic_candidate(text, parse_fn),
        "parse_result": verification.get("candidate_parsed"),
        "typed_verifier_evidence": verification,
        "math_verify_result": bool(verification.get("equivalent")),
        "is_correct": bool(verification.get("equivalent")),
        "finish_reason": finish_reason,
        "stop_token_observed": stop_token_id,
        "cap_status": bool(cap_hit),
        "think_closure": think_closed,
        "answer_replacement_count": sum(event.superseded for event in all_events),
        "partial_cap_suppression_count": partial_cap_suppressions,
        "structural_candidate_rejection_count": structural_rejections,
        "parser_verifier_exception": verification.get("exception") or extraction_exception,
        "parser_verifier_timeout": bool(verification.get("timeout")),
        "generated_text_sha256": hashlib.sha256(str(generated_text).encode()).hexdigest(),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--problem-id", default="manual")
    parser.add_argument("--cap-hit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(score_response(args.response, args.gold, finish_reason="length" if args.cap_hit else "stop", stop_token_id=None, cap_hit=args.cap_hit, problem_id=args.problem_id), indent=2))


if __name__ == "__main__":
    main()
