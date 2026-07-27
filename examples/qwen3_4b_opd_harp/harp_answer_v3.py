#!/usr/bin/env python3
"""Auditable answer-event extraction and typed HARP verification, version 3.

The official HARP checker is deliberately evidence-only in this scorer.  A
positive grade requires a successful typed comparison; parser failures never
compare equal and are surfaced as ``needs_review``.
"""

from __future__ import annotations

import hashlib
import json
import re
import signal
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
from typing import Any, Callable, Iterator

from harp_official.latex_answer_check import check_one_latex_answer


SCORER_VERSION = "harp_answer_v3"
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
ANSWER_MARKER = re.compile(
    r"(?im)(?:final\s+(?:answer|result)\s*(?:is|:|=)?|"
    r"(?:the\s+)?answer\s+(?:is|equals)|answer\s*:|"
    r"my\s+(?:final\s+)?answer\s*(?:is|:|=)?)\s*"
)
CONCLUSION_MARKER = re.compile(r"(?im)\b(?:therefore|thus|hence|consequently)\b\s*[,;:]?\s*")
RETRACTION_MARKER = re.compile(
    r"(?i)\b(?:"
    r"(?:that|this|the|my|our)\s+(?:(?:last|previous|final)\s+)?(?:answer|result)\s+is\s+(?:wrong|incorrect|false)|"
    r"i\s+(?:was|am)\s+wrong|"
    r"i\s+(?:retract|reject|discard)\s+(?:that|this|the\s+answer|my\s+answer)|"
    r"(?:reject|discard)\s+(?:that|this|the\s+answer|the\s+result)"
    r")\b"
)
UNCERTAINTY_TAIL = re.compile(
    r"(?is)(?:[.!?;]\s*|,\s*)(?:but\s+)?(?:i\s+need\s+to\s+check|"
    r"let\s+me\s+check|i\s+am\s+not\s+sure|i'?m\s+not\s+sure|"
    r"perhaps|maybe|however|although|this\s+seems|wait\b).*$"
)


class VerificationTimeout(TimeoutError):
    pass


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


@contextmanager
def wall_clock_timeout(seconds: int = 10) -> Iterator[None]:
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def handler(_signum: int, _frame: Any) -> None:
        raise VerificationTimeout(f"typed verification exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def bounded_call(function: Callable[[], Any], seconds: int = 10) -> tuple[Any | None, dict[str, str] | None]:
    try:
        with wall_clock_timeout(seconds):
            return function(), None
    except Exception as exc:
        return None, {"type": type(exc).__name__, "message": str(exc)}


def clean_generated_text(generated_text: str | None) -> str:
    text = str(generated_text or "")
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def answer_region(text: str, *, cap_hit: bool) -> dict[str, Any]:
    """Return the only region eligible for answer selection.

    Cap hits use the unfinished trace.  Natural stops use the final region
    after ``</think>``.  Truly tagless responses use the whole response, while
    a naturally stopped unclosed thinking block has no final-answer region.
    """

    last_open = text.rfind("<think>")
    last_close = text.rfind("</think>")
    if cap_hit:
        status = "unclosed" if last_open > last_close else "closed" if last_close >= 0 else "tagless"
        return {"text": text, "start": 0, "end": len(text), "think_status": status, "region": "unfinished_trace"}
    if last_open > last_close:
        return {"text": "", "start": len(text), "end": len(text), "think_status": "unclosed", "region": "missing_post_think"}
    if last_close >= 0:
        start = last_close + len("</think>")
        return {"text": text[start:], "start": start, "end": len(text), "think_status": "closed", "region": "post_think"}
    return {
        "text": text,
        "start": 0,
        "end": len(text),
        "think_status": "tagless" if text.strip() else "missing",
        "region": "tagless_final",
    }


def scan_complete_boxes(text: str, *, absolute_offset: int = 0) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    search_position = 0
    pattern = re.compile(r"\\(boxed|fbox)\s*\{")
    while True:
        match = pattern.search(text, search_position)
        if match is None:
            break
        open_brace = text.find("{", match.start())
        depth = 0
        close_brace: int | None = None
        index = open_brace
        while index < len(text):
            char = text[index]
            slash_count = 0
            cursor = index - 1
            while cursor >= 0 and text[cursor] == "\\":
                slash_count += 1
                cursor -= 1
            escaped = slash_count % 2 == 1
            if not escaped:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        close_brace = index
                        break
            index += 1
        if close_brace is None:
            search_position = match.end()
            continue
        boxes.append(
            {
                "command": match.group(1),
                "start": absolute_offset + match.start(),
                "end": absolute_offset + close_brace + 1,
                "local_start": match.start(),
                "local_end": close_brace + 1,
                "content": text[open_brace + 1 : close_brace].strip(),
                "raw": text[match.start() : close_brace + 1],
            }
        )
        search_position = close_brace + 1
    return boxes


def collection_separator(text: str) -> bool:
    return bool(re.fullmatch(r"[\s,;:.&]*(?:(?:and|or)\b[\s,;:.&]*)?", text, flags=re.IGNORECASE))


def trim_marker_candidate(value: str) -> str:
    candidate = value.strip()
    candidate = UNCERTAINTY_TAIL.sub("", candidate).strip()
    candidate = re.sub(
        r"(?is)^(.+?),\s*(?:based\s+on|according\s+to|because|since|which\s+(?:means|shows)|as\s+(?:computed|calculated)).*$",
        r"\1",
        candidate,
    ).strip()
    candidate = re.sub(r"\s*(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", candidate).strip()
    return candidate


def marker_candidate_end(text: str, start: int) -> int:
    line_end = text.find("\n", start)
    return len(text) if line_end < 0 else line_end


def make_event(
    *,
    kind: str,
    start: int,
    end: int,
    raw: str,
    candidate: str | None,
    components: list[str] | None = None,
    strength: str,
    eligible: bool,
    region: str,
    member_event_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": "",
        "kind": kind,
        "span": {"start": start, "end": end},
        "raw": raw,
        "candidate": candidate,
        "components": list(components or []),
        "strength": strength,
        "eligible": eligible,
        "region": region,
        "member_event_ids": list(member_event_ids or []),
    }


def concise_standalone_candidate(region_text: str) -> tuple[str, int, int] | None:
    leading = len(region_text) - len(region_text.lstrip())
    value = region_text.strip()
    if not value or len(value) > 256 or len(re.findall(r"\S+", value)) > 12:
        return None
    if len(re.findall(r"[.!?](?:\s|$)", value)) > 1 or "\n\n" in value:
        return None
    if re.search(r"(?i)\b(?:because|suppose|calculate|then|next|checking|maybe|perhaps|however|let\s+us)\b", value):
        return None
    return value, leading, leading + len(value)


def extract_answer_events(generated_text: str | None, *, cap_hit: bool) -> dict[str, Any]:
    text = clean_generated_text(generated_text)
    region = answer_region(text, cap_hit=cap_hit)
    region_text = str(region["text"])
    offset = int(region["start"])
    events: list[dict[str, Any]] = []

    boxes = scan_complete_boxes(region_text, absolute_offset=offset)
    box_events: list[dict[str, Any]] = []
    for box in boxes:
        event = make_event(
            kind="balanced_box",
            start=box["start"],
            end=box["end"],
            raw=box["raw"],
            candidate=box["content"] or None,
            components=[box["content"]] if box["content"] else [],
            strength="strong",
            eligible=True,
            region=region["region"],
        )
        box_events.append(event)
        events.append(event)

    # Adjacent boxes form one multipart answer.  Member boxes remain in the
    # audit but are not separately eligible for selection.
    index = 0
    while index < len(boxes):
        run_end = index
        while run_end + 1 < len(boxes):
            separator = region_text[boxes[run_end]["local_end"] : boxes[run_end + 1]["local_start"]]
            if not collection_separator(separator):
                break
            run_end += 1
        if run_end > index:
            members = box_events[index : run_end + 1]
            for member in members:
                member["eligible"] = False
                member["selection_note"] = "member_of_adjacent_box_collection"
            components = [str(member["candidate"]) for member in members]
            events.append(
                make_event(
                    kind="adjacent_box_collection",
                    start=members[0]["span"]["start"],
                    end=members[-1]["span"]["end"],
                    raw=text[members[0]["span"]["start"] : members[-1]["span"]["end"]],
                    candidate=", ".join(components),
                    components=components,
                    strength="strong",
                    eligible=True,
                    region=region["region"],
                )
            )
        index = run_end + 1

    for match in ANSWER_MARKER.finditer(region_text):
        local_end = marker_candidate_end(region_text, match.end())
        raw_candidate = region_text[match.end() : local_end]
        candidate = trim_marker_candidate(raw_candidate)
        candidate_end = match.end() + len(raw_candidate.rstrip())
        events.append(
            make_event(
                kind="explicit_final_marker",
                start=offset + match.start(),
                end=offset + max(match.end(), candidate_end),
                raw=region_text[match.start() : max(match.end(), candidate_end)],
                candidate=candidate or None,
                strength="strong",
                eligible=bool(candidate),
                region=region["region"],
            )
        )

    # Generic conclusions are deliberately audit-only, especially on cap
    # hits; they do not become answers merely because they contain numbers.
    for match in CONCLUSION_MARKER.finditer(region_text):
        local_end = marker_candidate_end(region_text, match.end())
        candidate = trim_marker_candidate(region_text[match.end() : local_end])
        events.append(
            make_event(
                kind="generic_conclusion",
                start=offset + match.start(),
                end=offset + local_end,
                raw=region_text[match.start() : local_end],
                candidate=candidate or None,
                strength="weak",
                eligible=False,
                region=region["region"],
            )
        )

    if not cap_hit and not any(event["eligible"] and event["strength"] == "strong" for event in events):
        concise = concise_standalone_candidate(region_text)
        if concise is not None:
            candidate, local_start, local_end = concise
            events.append(
                make_event(
                    kind="concise_post_think",
                    start=offset + local_start,
                    end=offset + local_end,
                    raw=candidate,
                    candidate=candidate,
                    strength="weak",
                    eligible=True,
                    region=region["region"],
                )
            )

    retractions: list[dict[str, Any]] = []
    for match in RETRACTION_MARKER.finditer(region_text):
        retractions.append(
            {
                "event_id": "",
                "kind": "explicit_retraction",
                "span": {"start": offset + match.start(), "end": offset + match.end()},
                "raw": match.group(0),
                "candidate": None,
                "components": [],
                "strength": "control",
                "eligible": False,
                "region": region["region"],
                "member_event_ids": [],
            }
        )
    events.extend(retractions)
    events.sort(key=lambda event: (int(event["span"]["start"]), int(event["span"]["end"]), event["kind"]))
    for event_index, event in enumerate(events):
        event["event_id"] = f"event_{event_index:04d}"

    # Fill collection membership after stable event IDs exist.
    for event in events:
        if event["kind"] != "adjacent_box_collection":
            continue
        event["member_event_ids"] = [
            other["event_id"]
            for other in events
            if other["kind"] == "balanced_box"
            and event["span"]["start"] <= other["span"]["start"]
            and other["span"]["end"] <= event["span"]["end"]
        ]

    return {
        "generated_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "think_status": region["think_status"],
        "selection_region": region["region"],
        "selection_region_span": {"start": region["start"], "end": region["end"]},
        "selection_region_sha256": hashlib.sha256(region_text.encode("utf-8")).hexdigest(),
        "answer_events": events,
    }


def strip_outer_math(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == "$" and text[-1] == "$":
        text = text[1:-1].strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    return text


def unwrap_text_commands(value: str) -> str:
    text = value
    pattern = re.compile(r"\\(?:text|textbf|textrm|mathrm|mathbf|operatorname|mbox)\s*\{([^{}]*)\}")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(r"\1", text)
    return text


def normalize_surface(value: str) -> str:
    text = strip_outer_math(value)
    text = text.replace("−", "-").replace("–", "-")
    text = text.replace("≤", r"\le").replace("≥", r"\ge").replace("≠", r"\ne")
    text = text.replace("∞", r"\infty").replace("π", r"\pi")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\textdollar", "$\\mathrm{USD}~")
    text = unwrap_text_commands(text)
    text = re.sub(r"\\(?:,|:|;|!|quad|qquad|hspace\s*\{[^{}]*\})", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .;!")


def normalize_identity(value: str) -> str:
    return re.sub(r"\s+", "", normalize_surface(value)).lower()


def parse_simple_number(value: str) -> Fraction | None:
    text = normalize_surface(value)
    text = re.sub(r"(?<=\d)[,{](?=\d{3}(?:\D|$))", "", text)
    text = text.replace("{,}", "").strip()
    text = re.sub(r"^\+", "", text)
    mixed = re.fullmatch(r"([+-]?\d+)\s*\\frac\s*\{([0-9]+)\}\s*\{([0-9]+)\}", text)
    if mixed:
        whole = int(mixed.group(1))
        part = Fraction(int(mixed.group(2)), int(mixed.group(3)))
        return Fraction(whole, 1) + (part if whole >= 0 else -part)
    mixed_slash = re.fullmatch(r"([+-]?\d+)\s+([0-9]+)\s*/\s*([0-9]+)", text)
    if mixed_slash:
        whole = int(mixed_slash.group(1))
        part = Fraction(int(mixed_slash.group(2)), int(mixed_slash.group(3)))
        return Fraction(whole, 1) + (part if whole >= 0 else -part)
    latex_fraction = re.fullmatch(r"\\frac\s*\{?([+-]?\d+(?:\.\d+)?)\}?\s*\{?([+-]?\d+(?:\.\d+)?)\}?", text)
    if latex_fraction:
        denominator = Fraction(latex_fraction.group(2))
        return Fraction(latex_fraction.group(1)) / denominator if denominator else None
    slash_fraction = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)", text)
    if slash_fraction:
        denominator = Fraction(slash_fraction.group(2))
        return Fraction(slash_fraction.group(1)) / denominator if denominator else None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return Fraction(text)
    return None


EMPTY_ALIASES = {
    "no root",
    "no roots",
    "no solution",
    "no solutions",
    "no real root",
    "no real roots",
    "no real solution",
    "no real solutions",
    "empty set",
    "the empty set",
    r"\emptyset",
    r"\varnothing",
    "∅",
    "{}",
    r"\{\}",
}
COLORS = {"red", "orange", "yellow", "green", "blue", "indigo", "violet", "purple", "black", "white", "gray", "grey"}
DIRECTIONS = {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "up", "down", "left", "right"}
WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
UNIT_ALIASES: dict[str, tuple[str, str, Fraction]] = {
    "in": ("length", "inch", Fraction(1)),
    "inch": ("length", "inch", Fraction(1)),
    "inches": ("length", "inch", Fraction(1)),
    "ft": ("length", "inch", Fraction(12)),
    "foot": ("length", "inch", Fraction(12)),
    "feet": ("length", "inch", Fraction(12)),
    "yd": ("length", "inch", Fraction(36)),
    "yard": ("length", "inch", Fraction(36)),
    "yards": ("length", "inch", Fraction(36)),
    "cm": ("length_metric", "cm", Fraction(1)),
    "centimeter": ("length_metric", "cm", Fraction(1)),
    "centimeters": ("length_metric", "cm", Fraction(1)),
    "m": ("length_metric", "cm", Fraction(100)),
    "meter": ("length_metric", "cm", Fraction(100)),
    "meters": ("length_metric", "cm", Fraction(100)),
    "sec": ("time", "second", Fraction(1)),
    "second": ("time", "second", Fraction(1)),
    "seconds": ("time", "second", Fraction(1)),
    "min": ("time", "second", Fraction(60)),
    "minute": ("time", "second", Fraction(60)),
    "minutes": ("time", "second", Fraction(60)),
    "hr": ("time", "second", Fraction(3600)),
    "hour": ("time", "second", Fraction(3600)),
    "hours": ("time", "second", Fraction(3600)),
    "degree": ("angle", "degree", Fraction(1)),
    "degrees": ("angle", "degree", Fraction(1)),
    "deg": ("angle", "degree", Fraction(1)),
    "mph": ("speed", "mph", Fraction(1)),
    "k": ("temperature", "kelvin", Fraction(1)),
    "kelvin": ("temperature", "kelvin", Fraction(1)),
}


def problem_supports_ratio(problem_text: str) -> bool:
    return bool(re.search(r"(?i)\b(?:ratio|rate|proportion|for every)\b", problem_text))


def problem_supports_dimensions(problem_text: str) -> bool:
    return bool(re.search(r"(?i)\b(?:dimensions?|length\s+and\s+width|rows?\s+by\s+columns?|rectangle|array|grid)\b", problem_text))


def problem_supplies_unit(problem_text: str, unit: str) -> bool:
    aliases = [name for name, metadata in UNIT_ALIASES.items() if metadata[0] == UNIT_ALIASES[unit][0]]
    return any(re.search(rf"(?i)\b{re.escape(alias)}\b", problem_text) for alias in aliases)


def split_top_level(value: str, pattern: re.Pattern[str]) -> list[str]:
    pieces: list[str] = []
    start = 0
    brace = paren = bracket = 0
    for match in pattern.finditer(value):
        prefix = value[start : match.start()]
        for char in prefix:
            if char == "{":
                brace += 1
            elif char == "}":
                brace = max(0, brace - 1)
            elif char == "(":
                paren += 1
            elif char == ")":
                paren = max(0, paren - 1)
            elif char == "[":
                bracket += 1
            elif char == "]":
                bracket = max(0, bracket - 1)
        if brace == 0 and paren == 0 and bracket == 0:
            pieces.append(value[start : match.start()].strip())
            start = match.end()
    pieces.append(value[start:].strip())
    return [piece for piece in pieces if piece]


def parsed_failure(raw: str, reason: str) -> dict[str, Any]:
    return {"success": False, "kind": "unparsed", "raw": raw, "surface": normalize_surface(raw), "reason": reason}


def parse_answer(value: str, problem_text: str, *, expected_kind: str | None = None) -> dict[str, Any]:
    raw = str(value)
    surface = normalize_surface(raw)
    lowered = surface.lower().strip()
    base: dict[str, Any] = {"success": True, "raw": raw, "surface": surface}
    if not surface:
        return parsed_failure(raw, "empty_candidate")

    if lowered in EMPTY_ALIASES:
        return {**base, "kind": "empty_solution", "canonical": "empty_solution"}
    if lowered in WORD_NUMBERS:
        number = Fraction(WORD_NUMBERS[lowered])
        return {**base, "kind": "scalar", "numeric": str(number), "canonical": str(number), "number_word": lowered}

    if re.fullmatch(r"(?:\\infty|\+\\infty|infinity|positive infinity)", lowered):
        return {**base, "kind": "infinity", "canonical": "+infinity"}
    if re.fullmatch(r"(?:-\\infty|-infinity|negative infinity)", lowered):
        return {**base, "kind": "infinity", "canonical": "-infinity"}

    # Ordered labeled directions preserve both order and direction labels.
    direction_matches = list(
        re.finditer(
            r"(?i)([+-]?(?:\d+(?:\.\d+)?|\\frac\s*\{\d+\}\s*\{\d+\}))\s*(north|south|east|west|northeast|northwest|southeast|southwest|up|down|left|right)\b",
            surface,
        )
    )
    if direction_matches:
        components = [
            {"magnitude": parse_simple_number(match.group(1)), "label": match.group(2).lower()}
            for match in direction_matches
        ]
        if all(component["magnitude"] is not None for component in components):
            return {
                **base,
                "kind": "labeled_directions",
                "ordered": True,
                "components": [
                    {"magnitude": str(component["magnitude"]), "label": component["label"]} for component in components
                ],
                "canonical": [
                    [component["label"], str(component["magnitude"])] for component in components
                ],
            }

    # Intervals are parsed before generic collections.
    interval = re.fullmatch(r"([\[(])\s*(.+?)\s*,\s*(.+?)\s*([\])])", surface)
    if interval:
        return {
            **base,
            "kind": "interval",
            "lower_closed": interval.group(1) == "[",
            "upper_closed": interval.group(4) == "]",
            "lower": parse_answer(interval.group(2), problem_text, expected_kind="scalar"),
            "upper": parse_answer(interval.group(3), problem_text, expected_kind="scalar"),
            "canonical": surface,
        }

    if re.search(r"(?:<=|>=|<|>|\\leq?|\\geq?|\\neq?|=)", surface) and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", surface):
        normalized = (
            surface.replace(r"\leq", r"\le")
            .replace(r"\geq", r"\ge")
            .replace("<=", r"\le")
            .replace(">=", r"\ge")
        )
        return {**base, "kind": "inequality_or_equation", "canonical": re.sub(r"\s+", "", normalized)}

    percent_match = re.fullmatch(r"(.+?)\s*(?:\\%|%)", surface)
    if percent_match:
        magnitude = parse_simple_number(percent_match.group(1))
        if magnitude is not None:
            return {
                **base,
                "kind": "percentage",
                "magnitude": str(magnitude),
                "canonical": str(magnitude / 100),
            }
        return {**base, "kind": "percentage", "canonical": surface, "symbolic": percent_match.group(1)}

    currency_match = re.search(r"(?i)(?:\$|USD\s*~?)\s*([+-]?\d+(?:\.\d+)?)|([+-]?\d+(?:\.\d+)?)\s*(?:dollars?|USD)\b", surface)
    if currency_match:
        magnitude_text = currency_match.group(1) or currency_match.group(2)
        magnitude = parse_simple_number(magnitude_text)
        if magnitude is not None and re.search(r"(?i)\b(?:loses?|lost|loss|owes?|decrease[sd]?)\b", surface[: currency_match.start()]):
            magnitude = -abs(magnitude)
        return {
            **base,
            "kind": "quantity",
            "dimension": "currency",
            "unit": "USD",
            "unit_power": 1,
            "magnitude": str(magnitude) if magnitude is not None else magnitude_text,
            "canonical": ["currency", "USD", str(magnitude) if magnitude is not None else magnitude_text],
        }

    degree_match = re.fullmatch(r"(.+?)\s*(?:\^\s*\{?\\circ\}?|°)", surface)
    if degree_match:
        magnitude = parse_simple_number(degree_match.group(1))
        return {
            **base,
            "kind": "quantity",
            "dimension": "angle",
            "unit": "degree",
            "unit_power": 1,
            "magnitude": str(magnitude) if magnitude is not None else degree_match.group(1),
            "canonical": ["angle", "degree", str(magnitude) if magnitude is not None else degree_match.group(1)],
        }

    unit_match = re.fullmatch(r"(.+?)\s+([A-Za-z]+)(?:\s*\^\s*\{?([+-]?\d+)\}?)?", surface)
    if unit_match and unit_match.group(2).lower() in UNIT_ALIASES:
        unit_key = unit_match.group(2).lower()
        dimension, base_unit, factor = UNIT_ALIASES[unit_key]
        power = int(unit_match.group(3) or "1")
        magnitude = parse_simple_number(unit_match.group(1))
        canonical_magnitude = magnitude * (factor**power) if magnitude is not None else unit_match.group(1)
        return {
            **base,
            "kind": "quantity",
            "dimension": dimension,
            "unit": unit_key,
            "base_unit": base_unit,
            "unit_power": power,
            "magnitude": str(magnitude) if magnitude is not None else unit_match.group(1),
            "canonical": [dimension, base_unit, power, str(canonical_magnitude)],
        }

    if expected_kind == "ratio" or ":" in surface or re.search(r"(?i)\bto\b", surface):
        ratio_parts = re.split(r"\s*(?::|\bto\b)\s*", surface, flags=re.IGNORECASE)
        if len(ratio_parts) != 2 and expected_kind == "ratio":
            ratio_parts = re.split(r"\s*/\s*", surface)
        if len(ratio_parts) != 2 and expected_kind == "ratio":
            ratio_value = parse_simple_number(surface)
            if ratio_value is not None:
                ratio_parts = [str(ratio_value.numerator), str(ratio_value.denominator)]
        if len(ratio_parts) == 2:
            left = parse_simple_number(ratio_parts[0])
            right = parse_simple_number(ratio_parts[1])
            if left is not None and right not in (None, Fraction(0)):
                return {
                    **base,
                    "kind": "ratio",
                    "components": [str(left), str(right)],
                    "canonical": str(left / right),
                    "ordered": True,
                }

    dimension_parts = re.split(r"\s*(?:\\times|×|\bby\b)\s*", surface, flags=re.IGNORECASE)
    if len(dimension_parts) == 2 and (expected_kind == "dimensions" or problem_supports_dimensions(problem_text)):
        parsed_parts = [parse_answer(part, problem_text, expected_kind="scalar") for part in dimension_parts]
        if all(part["success"] for part in parsed_parts):
            return {
                **base,
                "kind": "dimensions",
                "components": parsed_parts,
                "canonical": [part.get("canonical") for part in parsed_parts],
                "ordered": True,
            }

    set_wrapped = (surface.startswith(r"\{") and surface.endswith(r"\}")) or (
        surface.startswith("{") and surface.endswith("}")
    )
    ordered_wrapped = surface.startswith("(") and surface.endswith(")")
    collection_surface = surface[2:-2] if surface.startswith(r"\{") and surface.endswith(r"\}") else surface[1:-1] if (set_wrapped or ordered_wrapped) else surface
    collection_pattern = re.compile(r"\s*(?:;|,(?!\d{3}(?:\D|$))|\band\b|\bor\b|\bonly\b)\s*", re.IGNORECASE)
    collection_parts = split_top_level(collection_surface, collection_pattern)
    collection_signaled = set_wrapped or ordered_wrapped or bool(re.search(r"(?i)\b(?:and|or|only|either)\b", surface))
    collection_signaled = collection_signaled or (len(collection_parts) > 1 and not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", surface))
    if expected_kind == "collection" or collection_signaled:
        if len(collection_parts) > 1:
            parts = [parse_answer(part, problem_text) for part in collection_parts]
            return {
                **base,
                "kind": "collection",
                "ordered": ordered_wrapped,
                "connector": "or" if re.search(r"(?i)\bor\b", surface) else "and" if re.search(r"(?i)\band\b", surface) else "comma",
                "components": parts,
                "cardinality": len(parts),
                "canonical": [part.get("canonical", part.get("surface")) for part in parts],
            }

    number = parse_simple_number(surface)
    if number is not None:
        return {**base, "kind": "scalar", "numeric": str(number), "canonical": str(number)}

    roman_parts = [part.strip().upper() for part in re.split(r"\s*(?:,|\band\b|\bor\b)\s*", surface, flags=re.IGNORECASE) if part.strip()]
    if roman_parts and all(re.fullmatch(r"[IVXLCDM]+", part) for part in roman_parts):
        if len(roman_parts) == 1:
            return {**base, "kind": "roman_label", "canonical": roman_parts[0]}
        return {
            **base,
            "kind": "collection",
            "ordered": False,
            "connector": "selection",
            "cardinality": len(roman_parts),
            "components": [{"success": True, "kind": "roman_label", "canonical": part, "surface": part, "raw": part} for part in roman_parts],
            "canonical": roman_parts,
        }

    if lowered in COLORS:
        return {**base, "kind": "color_label", "canonical": lowered}
    if lowered in DIRECTIONS:
        return {**base, "kind": "direction_label", "canonical": lowered}

    # Text labels are intentionally whole-answer labels.  No letter-only
    # canonicalizer is used; coefficients, subscripts, and operators stay in
    # symbolic expressions instead of collapsing to a shared letter.
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", surface):
        words = re.sub(r"\s+", " ", surface).strip()
        if len(words.split()) <= 12:
            kind = "point_label" if re.fullmatch(r"[A-Z](?:_[A-Za-z0-9]+)?", words) else "text_label"
            return {**base, "kind": kind, "canonical": words.lower() if kind == "text_label" else words}

    prose_words = re.findall(r"[A-Za-z]+", surface)
    if len(prose_words) >= 3 and not re.search(r"\\(?:frac|sqrt|sum|prod|log|sin|cos|tan)|[=<>^]", surface):
        prose = re.sub(r"\s+", " ", surface).strip().lower()
        return {**base, "kind": "text_label", "canonical": prose}

    # Remaining mathematical syntax is a typed symbolic scalar.  The exact
    # surface is retained and equivalence requires both parsers to succeed.
    if re.search(r"[0-9A-Za-z\\^_+*/(){}-]", surface):
        return {**base, "kind": "scalar", "symbolic": surface, "canonical": re.sub(r"\s+", "", surface)}
    return parsed_failure(raw, "unsupported_answer_type")


def official_evidence(candidate: str, gold: str) -> dict[str, Any]:
    output, error = bounded_call(
        lambda: check_one_latex_answer(candidate, gold, extract_policy="flex", eval_policy="aggressive")
    )
    return {"output": json_safe(output), "error": error, "authoritative": False}


def math_verify_evidence(candidate: str, gold: str) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        from math_verify import parse, verify

        gold_input = gold if str(gold).strip().startswith("$") else f"${gold}$"
        candidate_input = candidate if str(candidate).strip().startswith("$") else f"${candidate}$"
        parsed_gold = parse(gold_input, parsing_timeout=10, raise_on_error=True)
        parsed_candidate = parse(candidate_input, parsing_timeout=10, raise_on_error=True)
        if not parsed_gold or not parsed_candidate:
            return {
                "parsed_gold": json_safe(parsed_gold),
                "parsed_candidate": json_safe(parsed_candidate),
                "parse_success": False,
                "equivalent": None,
            }
        return {
            "parsed_gold": json_safe(parsed_gold),
            "parsed_candidate": json_safe(parsed_candidate),
            "parse_success": True,
            "equivalent": bool(verify(parsed_gold, parsed_candidate, timeout_seconds=10, raise_on_error=True)),
        }

    result, error = bounded_call(run)
    if error is not None:
        return {"parse_success": False, "equivalent": None, "error": error}
    assert isinstance(result, dict)
    return {**result, "error": None}


def decision(grade: str, reason: str, **evidence: Any) -> dict[str, Any]:
    return {
        "grade": grade,
        "is_correct": True if grade == "correct" else False if grade == "incorrect" else None,
        "decision_reason": reason,
        **evidence,
    }


def compare_typed(candidate: dict[str, Any], gold: dict[str, Any], problem_text: str) -> dict[str, Any]:
    if not gold.get("success"):
        return decision("needs_review", "gold_parse_failed")
    if not candidate.get("success"):
        return decision("needs_review", "candidate_parse_failed")

    candidate_kind = str(candidate["kind"])
    gold_kind = str(gold["kind"])
    compatible_scalar_kinds = {"scalar", "percentage"}
    if candidate_kind != gold_kind and {candidate_kind, gold_kind} <= compatible_scalar_kinds:
        pass
    elif candidate_kind != gold_kind:
        if gold_kind == "inequality_or_equation" and candidate_kind == "scalar" and "=" in gold["surface"]:
            left, right = gold["surface"].split("=", 1)
            if re.fullmatch(r"\s*[A-Za-z](?:_[A-Za-z0-9]+)?\s*", left):
                rhs = parse_answer(right, problem_text, expected_kind="scalar")
                nested = compare_typed(candidate, rhs, problem_text)
                nested["assignment_lhs"] = left.strip()
                nested["decision_reason"] = f"assignment_rhs_{nested['decision_reason']}"
                return nested
        if candidate_kind == "inequality_or_equation" and gold_kind == "scalar" and "=" in candidate["surface"]:
            left, right = candidate["surface"].split("=", 1)
            if re.fullmatch(r"\s*[A-Za-z](?:_[A-Za-z0-9]+)?\s*", left):
                rhs = parse_answer(right, problem_text, expected_kind="scalar")
                nested = compare_typed(rhs, gold, problem_text)
                nested["assignment_lhs"] = left.strip()
                nested["decision_reason"] = f"assignment_rhs_{nested['decision_reason']}"
                return nested
        if gold_kind == "quantity" and candidate_kind == "scalar":
            magnitude_gold = parse_answer(str(gold.get("magnitude", "")), problem_text, expected_kind="scalar")
            nested = compare_typed(candidate, magnitude_gold, problem_text)
            nested["unit_omission_accepted"] = True
            nested["decision_reason"] = f"unit_omission_{nested['decision_reason']}"
            return nested
        if gold_kind == "ratio" and candidate_kind == "scalar" and problem_supports_ratio(problem_text):
            reparsed = parse_answer(candidate["raw"], problem_text, expected_kind="ratio")
            if reparsed.get("kind") == "ratio":
                return compare_typed(reparsed, gold, problem_text)
        return decision("incorrect", "incompatible_answer_types")

    if gold_kind in {"empty_solution", "infinity", "color_label", "direction_label", "roman_label", "point_label", "text_label"}:
        equal = candidate.get("canonical") == gold.get("canonical")
        return decision("correct" if equal else "incorrect", "exact_typed_label_match" if equal else "typed_label_mismatch")

    if gold_kind == "labeled_directions":
        equal = candidate.get("canonical") == gold.get("canonical")
        return decision("correct" if equal else "incorrect", "ordered_direction_match" if equal else "ordered_direction_mismatch")

    if gold_kind == "percentage":
        candidate_value = Fraction(str(candidate["canonical"])) if candidate_kind == "percentage" else parse_simple_number(candidate["surface"])
        gold_value = Fraction(str(gold["canonical"]))
        gold_magnitude = Fraction(str(gold["magnitude"]))
        if candidate_value is None:
            return decision("needs_review", "percentage_candidate_not_numeric")
        equal = candidate_value == gold_value or (candidate_kind == "scalar" and candidate_value == gold_magnitude)
        return decision("correct" if equal else "incorrect", "percentage_value_match" if equal else "percentage_value_mismatch")
    if gold_kind == "scalar" and candidate_kind == "percentage":
        candidate_value = Fraction(str(candidate["canonical"]))
        gold_value = parse_simple_number(gold["surface"])
        if gold_value is None:
            return decision("needs_review", "scalar_gold_not_numeric_for_percentage")
        equal = candidate_value == gold_value
        return decision("correct" if equal else "incorrect", "percentage_to_scalar_match" if equal else "percentage_to_scalar_mismatch")

    if gold_kind == "quantity":
        if candidate.get("dimension") != gold.get("dimension") or candidate.get("unit_power") != gold.get("unit_power"):
            return decision("incorrect", "conflicting_units")
        if candidate.get("base_unit", candidate.get("unit")) != gold.get("base_unit", gold.get("unit")):
            return decision("incorrect", "incompatible_units")
        candidate_canonical = candidate.get("canonical")
        gold_canonical = gold.get("canonical")
        equal = candidate_canonical == gold_canonical
        if equal:
            return decision("correct", "compatible_quantity_match")
        candidate_magnitude = parse_simple_number(str(candidate.get("magnitude", "")))
        gold_magnitude = parse_simple_number(str(gold.get("magnitude", "")))
        if candidate_magnitude is not None and gold_magnitude is not None:
            return decision("incorrect", "quantity_magnitude_mismatch")
        symbolic = math_verify_evidence(str(candidate.get("magnitude", "")), str(gold.get("magnitude", "")))
        if not symbolic.get("parse_success"):
            return decision("needs_review", "quantity_symbolic_parse_failed", math_verify=symbolic)
        return decision("correct" if symbolic["equivalent"] else "incorrect", "quantity_symbolic_comparison", math_verify=symbolic)

    if gold_kind == "ratio":
        equal = candidate.get("canonical") == gold.get("canonical")
        return decision("correct" if equal else "incorrect", "ordered_ratio_match" if equal else "ordered_ratio_mismatch")

    if gold_kind == "dimensions":
        candidate_components = candidate.get("components", [])
        gold_components = gold.get("components", [])
        if len(candidate_components) != len(gold_components):
            return decision("incorrect", "dimension_cardinality_mismatch")
        comparisons = [compare_typed(left, right, problem_text) for left, right in zip(candidate_components, gold_components, strict=True)]
        if any(item["grade"] == "needs_review" for item in comparisons):
            return decision("needs_review", "dimension_component_needs_review", component_decisions=comparisons)
        correct = all(item["grade"] == "correct" for item in comparisons)
        return decision("correct" if correct else "incorrect", "ordered_dimension_comparison", component_decisions=comparisons)

    if gold_kind == "collection":
        candidate_components = list(candidate.get("components", []))
        gold_components = list(gold.get("components", []))
        if len(candidate_components) != len(gold_components):
            return decision("incorrect", "collection_cardinality_mismatch")
        if bool(candidate.get("ordered")) != bool(gold.get("ordered")):
            return decision("incorrect", "collection_order_semantics_mismatch")
        audit: list[dict[str, Any]] = []
        if gold.get("ordered"):
            comparisons = [compare_typed(left, right, problem_text) for left, right in zip(candidate_components, gold_components, strict=True)]
            audit.extend({"candidate_index": i, "gold_index": i, "decision": item} for i, item in enumerate(comparisons))
            if any(item["grade"] == "needs_review" for item in comparisons):
                return decision("needs_review", "ordered_collection_needs_review", component_decisions=audit)
            correct = all(item["grade"] == "correct" for item in comparisons)
            return decision("correct" if correct else "incorrect", "ordered_collection_comparison", component_decisions=audit)

        cache: dict[tuple[int, int], dict[str, Any]] = {}

        def pair(left_index: int, right_index: int) -> dict[str, Any]:
            key = (left_index, right_index)
            if key not in cache:
                cache[key] = compare_typed(candidate_components[left_index], gold_components[right_index], problem_text)
                audit.append({"candidate_index": left_index, "gold_index": right_index, "decision": cache[key]})
            return cache[key]

        def match(left_index: int, used: set[int]) -> bool:
            if left_index == len(candidate_components):
                return True
            for right_index in range(len(gold_components)):
                if right_index in used:
                    continue
                if pair(left_index, right_index)["grade"] == "correct" and match(left_index + 1, used | {right_index}):
                    return True
            return False

        matched = match(0, set())
        if matched:
            return decision("correct", "unordered_collection_exact_match", component_decisions=audit)
        if any(item["decision"]["grade"] == "needs_review" for item in audit):
            return decision("needs_review", "unordered_collection_needs_review", component_decisions=audit)
        return decision("incorrect", "unordered_collection_mismatch", component_decisions=audit)

    if gold_kind == "interval":
        if candidate_kind == "interval":
            if candidate.get("lower_closed") != gold.get("lower_closed") or candidate.get("upper_closed") != gold.get("upper_closed"):
                return decision("incorrect", "interval_endpoint_openness_mismatch")
            lower = compare_typed(candidate["lower"], gold["lower"], problem_text)
            upper = compare_typed(candidate["upper"], gold["upper"], problem_text)
            if "needs_review" in {lower["grade"], upper["grade"]}:
                return decision("needs_review", "interval_endpoint_needs_review", endpoint_decisions=[lower, upper])
            correct = lower["grade"] == upper["grade"] == "correct"
            return decision("correct" if correct else "incorrect", "interval_endpoint_comparison", endpoint_decisions=[lower, upper])

    # Scalars, equations, and inequalities require exact successful parser
    # evidence.  Identical canonical syntax and exact rational values are safe
    # fast paths; all other equivalence is delegated to math_verify only after
    # both parses succeed.
    if candidate.get("numeric") is not None and gold.get("numeric") is not None:
        equal = candidate["numeric"] == gold["numeric"]
        return decision("correct" if equal else "incorrect", "exact_rational_match" if equal else "exact_rational_mismatch")
    if candidate.get("canonical") == gold.get("canonical"):
        return decision("correct", "identical_typed_canonical_form")
    symbolic = math_verify_evidence(candidate["surface"], gold["surface"])
    if not symbolic.get("parse_success"):
        return decision("needs_review", "typed_symbolic_parse_failed", math_verify=symbolic)
    return decision(
        "correct" if symbolic["equivalent"] else "incorrect",
        "typed_symbolic_equivalent" if symbolic["equivalent"] else "typed_symbolic_mismatch",
        math_verify=symbolic,
    )


def lightweight_equivalent(left: str, right: str, problem_text: str) -> bool:
    if normalize_identity(left) == normalize_identity(right):
        return True
    parsed_left = parse_answer(left, problem_text)
    parsed_right = parse_answer(right, problem_text, expected_kind=parsed_left.get("kind"))
    if not parsed_left.get("success") or not parsed_right.get("success"):
        return False
    if parsed_left.get("kind") == parsed_right.get("kind") and parsed_left.get("canonical") == parsed_right.get("canonical"):
        return True
    left_number = parse_simple_number(left)
    right_number = parse_simple_number(right)
    return left_number is not None and left_number == right_number


def select_answer(events: list[dict[str, Any]], problem_text: str) -> dict[str, Any]:
    timeline: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    retracted = False
    for event in events:
        if event["kind"] == "explicit_retraction":
            if selected is not None:
                retracted = True
                timeline.append(
                    {
                        "action": "retraction",
                        "event_id": event["event_id"],
                        "retracted_event_id": selected["event_id"],
                    }
                )
            else:
                timeline.append({"action": "ignored_retraction_without_answer", "event_id": event["event_id"]})
            continue
        if not event.get("eligible") or not event.get("candidate"):
            if event["kind"] == "generic_conclusion":
                timeline.append({"action": "ignored_generic_conclusion", "event_id": event["event_id"]})
            continue
        if selected is None:
            selected = event
            retracted = False
            timeline.append({"action": "initial_selection", "event_id": event["event_id"]})
            continue
        if retracted:
            timeline.append(
                {"action": "replacement_after_retraction", "event_id": event["event_id"], "replaced_event_id": selected["event_id"]}
            )
            selected = event
            retracted = False
            continue
        if lightweight_equivalent(str(selected["candidate"]), str(event["candidate"]), problem_text):
            timeline.append(
                {"action": "reaffirmation", "event_id": event["event_id"], "reaffirmed_event_id": selected["event_id"]}
            )
            selected = event
        else:
            timeline.append(
                {"action": "replacement", "event_id": event["event_id"], "replaced_event_id": selected["event_id"]}
            )
            selected = event
            retracted = False
    return {"selected_event": selected, "selection_trace": timeline, "unresolved_retraction": selected is not None and retracted}


def score_answer(
    generated_text: str | None,
    gold_answer: str,
    problem_text: str,
    problem_id: str,
    cap_hit: bool,
    finish_reason: str | None,
) -> dict[str, Any]:
    extraction = extract_answer_events(generated_text, cap_hit=bool(cap_hit))
    selection = select_answer(extraction["answer_events"], str(problem_text))
    selected = selection["selected_event"]
    base = {
        "scorer_version": SCORER_VERSION,
        "problem_id": str(problem_id),
        "cap_hit": bool(cap_hit),
        "finish_reason": finish_reason,
        **extraction,
        "selection_trace": selection["selection_trace"],
        "selected_event_id": selected["event_id"] if selected is not None else None,
        "candidate": selected.get("candidate") if selected is not None else None,
        "components": selected.get("components", []) if selected is not None else [],
        "extraction_method": selected.get("kind", "none") if selected is not None else "none",
        "replacement_count": sum(item["action"].startswith("replacement") for item in selection["selection_trace"]),
        "reaffirmation_count": sum(item["action"] == "reaffirmation" for item in selection["selection_trace"]),
        "retraction_count": sum(item["action"] == "retraction" for item in selection["selection_trace"]),
    }
    if selected is None:
        return {
            **base,
            "grade": "incorrect",
            "is_correct": False,
            "decision_reason": "no_eligible_answer_event",
            "parse_failure": True,
            "verifier_method": "not_run",
            "verifier_evidence": None,
        }
    if selection["unresolved_retraction"]:
        return {
            **base,
            "grade": "incorrect",
            "is_correct": False,
            "decision_reason": "explicit_retraction_without_replacement",
            "parse_failure": False,
            "verifier_method": "not_run_retracted",
            "verifier_evidence": None,
        }

    candidate_text = str(selected["candidate"])
    official = official_evidence(candidate_text, str(gold_answer))
    gold_parsed = parse_answer(str(gold_answer), str(problem_text))
    expected_kind = str(gold_parsed.get("kind")) if gold_parsed.get("success") else None
    if expected_kind == "ratio" and "/" in normalize_surface(candidate_text):
        candidate_parsed = parse_answer(candidate_text, str(problem_text), expected_kind="ratio")
    elif expected_kind == "dimensions":
        candidate_parsed = parse_answer(candidate_text, str(problem_text), expected_kind="dimensions")
    elif expected_kind == "collection" and selected.get("components"):
        candidate_parsed = {
            "success": True,
            "kind": "collection",
            "raw": candidate_text,
            "surface": normalize_surface(candidate_text),
            "ordered": bool(gold_parsed.get("ordered")),
            "connector": "adjacent_boxes",
            "cardinality": len(selected["components"]),
            "components": [parse_answer(component, str(problem_text)) for component in selected["components"]],
            "canonical": [normalize_surface(component) for component in selected["components"]],
        }
    else:
        candidate_parsed = parse_answer(candidate_text, str(problem_text), expected_kind=expected_kind)
    typed_decision = compare_typed(candidate_parsed, gold_parsed, str(problem_text))
    verifier_evidence = {
        "candidate_parse": candidate_parsed,
        "gold_parse": gold_parsed,
        "typed_decision": typed_decision,
        "official_checker": official,
    }
    return {
        **base,
        "grade": typed_decision["grade"],
        "is_correct": typed_decision["is_correct"],
        "decision_reason": typed_decision["decision_reason"],
        "parse_failure": not candidate_parsed.get("success") or not gold_parsed.get("success"),
        "verifier_method": f"typed_{gold_parsed.get('kind', 'unparsed')}",
        "verifier_evidence": verifier_evidence,
    }


def metrics_from_predictions(predictions: list[dict[str, Any]], *, refuse_reviews: bool = True) -> dict[str, Any]:
    reviews = [row for row in predictions if row.get("grade") == "needs_review"]
    if refuse_reviews and reviews:
        raise ValueError(f"Refusing to publish HARP V3 metrics with {len(reviews)} included needs_review rows")
    total = len(predictions)
    correct = sum(row.get("grade") == "correct" for row in predictions)
    cap_rows = [row for row in predictions if bool(row.get("cap_hit"))]
    non_cap_rows = [row for row in predictions if not bool(row.get("cap_hit"))]
    action_counts = Counter(
        action["action"]
        for row in predictions
        for action in row.get("selection_trace", [])
    )
    return {
        "scorer_version": SCORER_VERSION,
        "publishable": not reviews,
        "num_eval_problems": total,
        "correct_count": correct,
        "incorrect_count": sum(row.get("grade") == "incorrect" for row in predictions),
        "review_count": len(reviews),
        "accuracy": correct / total if total else 0.0,
        "cap_hit_count": len(cap_rows),
        "cap_hit_rate": len(cap_rows) / total if total else 0.0,
        "cap_hit_accuracy": sum(row.get("grade") == "correct" for row in cap_rows) / len(cap_rows) if cap_rows else 0.0,
        "non_cap_accuracy": sum(row.get("grade") == "correct" for row in non_cap_rows) / len(non_cap_rows) if non_cap_rows else 0.0,
        "parse_failures": sum(bool(row.get("parse_failure")) for row in predictions),
        "average_generated_length": sum(int(row.get("generated_token_count", 0)) for row in predictions) / total if total else 0.0,
        "extraction_method_counts": dict(sorted(Counter(row.get("extraction_method", "none") for row in predictions).items())),
        "decision_reason_counts": dict(sorted(Counter(row.get("decision_reason", "missing") for row in predictions).items())),
        "verifier_method_counts": dict(sorted(Counter(row.get("verifier_method", "missing") for row in predictions).items())),
        "answer_event_type_counts": dict(
            sorted(Counter(event["kind"] for row in predictions for event in row.get("answer_events", [])).items())
        ),
        "selection_action_counts": dict(sorted(action_counts.items())),
        "cap_hit_replacement_count": sum(int(row.get("replacement_count", 0)) for row in cap_rows),
        "cap_hit_reaffirmation_count": sum(int(row.get("reaffirmation_count", 0)) for row in cap_rows),
        "cap_hit_retraction_count": sum(int(row.get("retraction_count", 0)) for row in cap_rows),
    }
