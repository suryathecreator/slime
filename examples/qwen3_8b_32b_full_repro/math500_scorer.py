#!/usr/bin/env python3
"""Versioned answer-event MATH-500 scorer using math_verify 0.9.0."""

from __future__ import annotations

import hashlib
import json
import re
import signal
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


SCORER_VERSION = "math500_event_scorer_v1"
THINK_CLOSE = "</think>"
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
ASSERTION_RE = re.compile(
    r"(?im)\b(?:final\s+(?:answer|result)\s*(?:is|:|=)?|(?:the\s+)?answer\s+(?:is|equals)|answer\s*:|my\s+final\s+answer\s*(?:is|:|=)?)\s*"
)
CONCLUSION_RE = re.compile(r"(?im)\b(?:therefore|hence|thus|consequently)\s*[,,:]?\s*")
CORRECTION_RE = re.compile(r"(?i)\b(?:wait|actually|scratch\s+that|correction|instead)\b")
RETRACTION_RE = re.compile(
    r"(?i)\b(?:i\s+(?:was|am)\s+wrong|(?:that|this|the|my|our)\s+(?:answer|result)\s+is\s+(?:wrong|incorrect|false)|i\s+(?:retract|reject|discard)\b|scratch\s+that|not\s+(?:the\s+)?(?:answer|result))"
)
DOUBT_RE = re.compile(r"(?i)\b(?:maybe|perhaps|possibly|i\s+(?:think|guess)|not\s+sure|uncertain|doubt(?:ful)?)\b")
NEGATION_RE = re.compile(r"(?i)\bnot\s+([^\n.;,]{1,128})")
MATHISH_RE = re.compile(r"(?:\\[A-Za-z]+|\d|[=+\-*/^<>]|\$|\(|\)|\[|\]|\{|\})")


class VerificationTimeout(TimeoutError):
    pass


@dataclass
class AnswerEvent:
    event_id: str
    event_type: str
    span: list[int]
    raw: str
    candidate: str | None
    confidence: str
    parseable: bool = False
    parse_result: str | None = None
    parser_exception: str | None = None
    parser_timeout: bool = False
    invalidated: bool = False
    retracted: bool = False
    doubtful: bool = False
    superseded: bool = False
    invalidation_reasons: list[str] = field(default_factory=list)
    candidate_sha256: str | None = None


def clean_text(text: str | None) -> str:
    value = str(text or "")
    for token in SPECIAL_TOKENS:
        value = value.replace(token, "")
    return value


def answer_region(text: str) -> tuple[str, int, str, bool]:
    if THINK_CLOSE in text:
        offset = text.rfind(THINK_CLOSE) + len(THINK_CLOSE)
        return text[offset:], offset, "final_post_think_region", True
    return text, 0, "unfinished_full_response", False


def balanced_boxes(text: str, offset: int) -> tuple[list[AnswerEvent], list[dict[str, Any]]]:
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
                confidence="confident" if candidate else "invalid",
            )
        )
    return events, malformed


def candidate_tail(text: str, start: int) -> tuple[str | None, int]:
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    raw_tail = text[start:line_end]
    tail = raw_tail.strip()
    if not tail:
        return None, start
    box_events, _ = balanced_boxes(tail, 0)
    if box_events and box_events[0].span[0] == 0:
        return box_events[0].candidate, start + box_events[0].span[1]
    tail = re.split(
        r"(?i)(?:\.(?=\s|$)|;\s*|</think>|<\|(?:im_end|endoftext)\|>|\s+(?:because|since|which\s+means|as\s+required)\b|\b(?:wait|actually|scratch\s+that)\b)",
        tail,
        maxsplit=1,
    )[0]
    tail = tail.rstrip(" .,:;!`*")
    if not tail or not MATHISH_RE.search(tail):
        return None, start
    relative = raw_tail.find(tail)
    if relative < 0:
        relative = len(raw_tail) - len(raw_tail.lstrip())
    return tail, start + relative + len(tail)


def asserted_events(text: str, offset: int) -> list[AnswerEvent]:
    events: list[AnswerEvent] = []
    for event_type, pattern in (("explicit_assertion", ASSERTION_RE), ("conclusion", CONCLUSION_RE)):
        for marker in pattern.finditer(text):
            candidate, end = candidate_tail(text, marker.end())
            if candidate is None:
                continue
            local = text[max(0, marker.start() - 32) : min(len(text), end + 32)]
            doubtful = bool(DOUBT_RE.search(local))
            events.append(
                AnswerEvent(
                    event_id="",
                    event_type=event_type,
                    span=[offset + marker.start(), offset + end],
                    raw=text[marker.start() : end],
                    candidate=candidate,
                    confidence="doubtful" if doubtful else "confident",
                    doubtful=doubtful,
                )
            )
    return events


def standalone_final_event(text: str, offset: int) -> AnswerEvent | None:
    lines = list(re.finditer(r"(?m)^.*$", text))
    nonempty = [line for line in lines if line.group(0).strip()]
    if not nonempty:
        return None
    line = nonempty[-1]
    candidate = line.group(0).strip().strip("$ ")
    if len(candidate) > 512 or not MATHISH_RE.search(candidate):
        return None
    strongly_indicated = bool(re.fullmatch(r"(?:\\boxed\{.*\}|[\[\](){}\\A-Za-z0-9_.,+\-*/^=<>|: ]+)", candidate))
    if not strongly_indicated:
        return None
    local = text[max(0, line.start() - 80) : line.end()]
    doubtful = bool(DOUBT_RE.search(local))
    return AnswerEvent(
        event_id="",
        event_type="standalone_final_math_line",
        span=[offset + line.start(), offset + line.end()],
        raw=line.group(0),
        candidate=candidate,
        confidence="doubtful" if doubtful else "confident",
        doubtful=doubtful,
    )


def parse_candidate(candidate: str | None, parse_fn: Callable[..., Any] | None = None) -> tuple[bool, str | None]:
    if not candidate:
        return False, None
    try:
        if parse_fn is None:
            from math_verify import parse as parse_fn
        parsed = parse_math(candidate, parse_fn=parse_fn, timeout_seconds=10)
        if parsed is None or parsed == []:
            return False, "unparseable_candidate"
        return True, repr(parsed)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def extract_events(region: str, offset: int, parse_fn: Callable[..., Any] | None = None) -> tuple[list[AnswerEvent], list[dict[str, Any]]]:
    boxes, malformed = balanced_boxes(region, offset)
    events = boxes + asserted_events(region, offset)
    weak = standalone_final_event(region, offset)
    if weak is not None and not any(
        event.span[0] < weak.span[1] and weak.span[0] < event.span[1] for event in events
    ):
        events.append(weak)
    events.sort(key=lambda event: (event.span[0], event.span[1], event.event_type))
    deduped: list[AnswerEvent] = []
    seen: set[tuple[int, int, str | None]] = set()
    for event in events:
        key = (event.span[0], event.span[1], event.candidate)
        if key in seen:
            continue
        seen.add(key)
        event.event_id = f"answer_{len(deduped):04d}"
        event.parseable, parse_detail = parse_candidate(event.candidate, parse_fn=parse_fn)
        if event.parseable:
            event.parse_result = parse_detail
        else:
            event.parser_exception = parse_detail
            event.parser_timeout = bool(parse_detail and "timeout" in parse_detail.lower())
        event.candidate_sha256 = hashlib.sha256(str(event.candidate or "").encode()).hexdigest()
        deduped.append(event)
    return deduped, malformed


def apply_discourse(text: str, events: list[AnswerEvent]) -> tuple[AnswerEvent | None, list[dict[str, Any]]]:
    controls: list[tuple[int, str, str]] = []
    for pattern, kind in ((CORRECTION_RE, "correction"), (RETRACTION_RE, "retraction"), (DOUBT_RE, "doubt")):
        controls.extend((match.start(), kind, match.group(0)) for match in pattern.finditer(text))
    controls.extend((match.start(), "negation", match.group(0)) for match in NEGATION_RE.finditer(text))
    controls.sort()
    timeline: list[tuple[int, str, Any]] = [(event.span[0], "answer", event) for event in events]
    timeline.extend((position, "control", (kind, raw)) for position, kind, raw in controls)
    timeline.sort(key=lambda item: (item[0], 0 if item[1] == "control" else 1))
    active: AnswerEvent | None = None
    history: list[dict[str, Any]] = []
    for position, kind, payload in timeline:
        if kind == "control":
            control, raw = payload
            if active is None:
                history.append({"action": f"ignored_{control}", "position": position, "raw": raw})
                continue
            if control == "doubt" and position <= active.span[1]:
                active.doubtful = True
                active.confidence = "doubtful"
                history.append({"action": "candidate_marked_doubtful", "event_id": active.event_id, "raw": raw})
            elif control == "doubt" and 0 <= position - active.span[1] <= 160:
                active.doubtful = True
                active.invalidated = True
                active.confidence = "doubtful"
                active.invalidation_reasons.append("later_tied_doubt")
                history.append({"action": "candidate_invalidated_by_doubt", "event_id": active.event_id, "raw": raw})
                active = None
            elif control in {"correction", "retraction"} and position >= active.span[1]:
                active.invalidated = True
                active.retracted = control == "retraction"
                active.invalidation_reasons.append(control)
                history.append({"action": f"candidate_{control}", "event_id": active.event_id, "raw": raw})
                active = None
            elif control == "negation" and position >= active.span[1]:
                normalized_candidate = re.sub(r"\s+", "", str(active.candidate or "")).lower().strip("$.")
                normalized_negation = re.sub(r"\s+", "", raw[3:]).lower().strip("$.")
                if normalized_candidate and normalized_negation.startswith(normalized_candidate):
                    active.invalidated = True
                    active.retracted = True
                    active.invalidation_reasons.append("tied_negation")
                    history.append({"action": "candidate_tied_negation", "event_id": active.event_id, "raw": raw})
                    active = None
            continue
        event: AnswerEvent = payload
        if event.doubtful or not event.parseable:
            history.append({"action": "ignored_doubtful_or_unparseable", "event_id": event.event_id})
            continue
        if active is not None:
            if active.candidate == event.candidate:
                history.append({"action": "reaffirmation", "event_id": event.event_id, "previous_event_id": active.event_id})
            else:
                active.superseded = True
                active.invalidated = True
                active.invalidation_reasons.append("later_confident_replacement")
                history.append({"action": "replacement", "event_id": event.event_id, "replaced_event_id": active.event_id})
        else:
            history.append({"action": "selection", "event_id": event.event_id})
        active = event
    if active is not None and (active.invalidated or active.doubtful or not active.parseable):
        active = None
    if active is None:
        for event in reversed(events):
            if event.parseable and not event.invalidated and not event.doubtful and event.confidence == "confident":
                active = event
                history.append({"action": "backward_last_confident_selection", "event_id": event.event_id})
                break
    return active, history


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise VerificationTimeout("math_verify timed out")


def parse_math(value: str, *, parse_fn: Callable[..., Any], timeout_seconds: int) -> Any:
    stripped = value.strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        attempts = [stripped]
    else:
        # Raw math such as 2\sqrt{2} is otherwise often reduced to its first
        # incidental number by the expression fallback. Prefer explicit LaTeX
        # math delimiters, then retain the raw extraction as a fallback.
        attempts = [f"${stripped}$", stripped]
    last: Any = []
    for attempt in attempts:
        last = parse_fn(attempt, parsing_timeout=min(10, timeout_seconds), raise_on_error=True)
        if last not in (None, []):
            return last
    return last


def verify_equivalence(candidate: str, gold: str, timeout_seconds: int = 15) -> dict[str, Any]:
    from math_verify import parse, verify

    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        parsed_gold = parse_math(gold, parse_fn=parse, timeout_seconds=timeout_seconds)
        parsed_candidate = parse_math(candidate, parse_fn=parse, timeout_seconds=timeout_seconds)
        if parsed_gold in (None, []) or parsed_candidate in (None, []):
            raise ValueError(f"unparseable math: gold={parsed_gold!r} candidate={parsed_candidate!r}")
        equivalent = bool(verify(parsed_gold, parsed_candidate))
        return {
            "parse_success": True,
            "gold_parsed": repr(parsed_gold),
            "candidate_parsed": repr(parsed_candidate),
            "equivalent": equivalent,
            "exception": None,
            "timeout": False,
        }
    except VerificationTimeout as exc:
        return {"parse_success": False, "gold_parsed": None, "candidate_parsed": None, "equivalent": False, "exception": str(exc), "timeout": True}
    except Exception as exc:
        return {"parse_success": False, "gold_parsed": None, "candidate_parsed": None, "equivalent": False, "exception": f"{type(exc).__name__}: {exc}", "timeout": False}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def diagnostic_candidate(text: str, parse_fn: Callable[..., Any] | None = None) -> str | None:
    events, _ = extract_events(text, 0, parse_fn=parse_fn)
    for event in reversed(events):
        if event.parseable and not event.doubtful:
            return event.candidate
    return None


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
    events, malformed_boxes = extract_events(region, offset, parse_fn=parse_fn)
    selected, history = apply_discourse(text, events)
    candidate = selected.candidate if selected is not None else None
    boxed = next((event.candidate for event in reversed(events) if event.event_type == "boxed" and event.parseable), None)
    last_confident = next((event.candidate for event in reversed(events) if event.parseable and not event.invalidated and not event.doubtful), None)
    anywhere = diagnostic_candidate(text, parse_fn=parse_fn)
    verification = {
        "parse_success": False,
        "gold_parsed": None,
        "candidate_parsed": None,
        "equivalent": False,
        "exception": None,
        "timeout": False,
    }
    if candidate is not None:
        verification = (verifier or verify_equivalence)(candidate, gold_answer)
    event_exception = next((event.parser_exception for event in reversed(events) if event.parser_exception), None)
    return {
        "scorer_version": SCORER_VERSION,
        "problem_id": str(problem_id),
        "official_extracted_candidate": candidate,
        "official_candidate_sha256": hashlib.sha256(str(candidate or "").encode()).hexdigest(),
        "extraction_region": region_name,
        "extraction_rule": selected.event_type if selected is not None else "no_confident_unretracted_candidate",
        "candidate_event_history": [asdict(event) for event in events],
        "selection_history": history,
        "invalidation_flags": {
            "any_retraction": any(event.retracted for event in events),
            "any_doubt": any(event.doubtful for event in events),
            "any_replacement": any(event.superseded for event in events),
            "malformed_boxes": malformed_boxes,
        },
        "boxed_answer_candidate": boxed,
        "last_confident_answer_candidate": last_confident,
        "answer_anywhere_diagnostic": anywhere,
        "parse_result": verification.get("candidate_parsed"),
        "math_verify_result": bool(verification.get("equivalent")),
        "is_correct": bool(verification.get("equivalent")),
        "finish_reason": finish_reason,
        "stop_token_observed": stop_token_id,
        "cap_status": bool(cap_hit),
        "think_closure": think_closed,
        "answer_replacement_count": sum(1 for event in events if event.superseded),
        "parser_verifier_exception": verification.get("exception") or event_exception,
        "parser_verifier_timeout": bool(verification.get("timeout")) or any(event.parser_timeout for event in events),
        "generated_text_sha256": hashlib.sha256(str(generated_text).encode()).hexdigest(),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--response", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--problem-id", default="manual")
    args = parser.parse_args()
    print(json.dumps(score_response(args.response, args.gold, finish_reason=None, stop_token_id=None, cap_hit=False, problem_id=args.problem_id), indent=2))


if __name__ == "__main__":
    main()
