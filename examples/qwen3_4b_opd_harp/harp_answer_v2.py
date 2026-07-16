#!/usr/bin/env python3
"""Auditable HARP answer extraction and bounded symbolic verification, version 2."""

from __future__ import annotations

import hashlib
import json
import re
import signal
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from harp_official.latex_answer_check import check_one_latex_answer


SCORER_VERSION = "harp_answer_v2"
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


class VerificationTimeout(TimeoutError):
    pass


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


@contextmanager
def wall_clock_timeout(seconds: int = 10) -> Iterator[None]:
    """Bound one symbolic verification attempt with SIGALRM."""
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def handler(_signum: int, _frame: Any) -> None:
        raise VerificationTimeout(f"symbolic verification exceeded {seconds} seconds")

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


def isolate_final_region(generated_text: str | None) -> tuple[str, str]:
    text = str(generated_text or "")
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    last_open = text.rfind("<think>")
    last_close = text.rfind("</think>")
    if last_open > last_close:
        return text[last_open + len("<think>") :].strip(), "unclosed"
    if last_close >= 0:
        return text[last_close + len("</think>") :].strip(), "closed"
    stripped = text.strip()
    return stripped, "tagless" if stripped else "missing"


def scan_complete_boxes(text: str) -> list[dict[str, Any]]:
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
            escaped = index > 0 and text[index - 1] == "\\" and (index < 2 or text[index - 2] != "\\")
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
                "start": match.start(),
                "end": close_brace + 1,
                "content": text[open_brace + 1 : close_brace].strip(),
            }
        )
        # A nested box belongs to the outer box content and must not become a
        # second top-level answer candidate.
        search_position = close_brace + 1
    return boxes


def collection_separator(text: str) -> bool:
    return bool(re.fullmatch(r"[\s,;:.&]*(?:(?:and|or)\b[\s,;:.&]*)?", text, flags=re.IGNORECASE))


def extract_candidate(post_think: str) -> dict[str, Any]:
    boxes = scan_complete_boxes(post_think)
    if boxes:
        selected = [boxes[-1]]
        cursor = len(boxes) - 1
        while cursor > 0 and collection_separator(post_think[boxes[cursor - 1]["end"] : boxes[cursor]["start"]]):
            selected.insert(0, boxes[cursor - 1])
            cursor -= 1
        components = [box["content"] for box in selected]
        return {
            "candidate": ", ".join(components),
            "components": components,
            "extraction_method": "box_collection" if len(selected) > 1 else selected[0]["command"],
            "box_spans": boxes,
        }

    explicit: list[tuple[int, str, str]] = []
    conclusions: list[tuple[int, str, str]] = []
    for match in re.finditer(
        r"(?im)(?:final\s+answer\s*(?:is|:|=)?|answer\s+is|answer\s*:)\s*(.+?)\s*$",
        post_think,
    ):
        explicit.append((match.start(), match.group(1).strip(), "explicit_marker"))
    for match in re.finditer(r"(?im)(?:therefore|thus|hence)\s*[,;:]?\s*(.+?)\s*$", post_think):
        conclusions.append((match.start(), match.group(1).strip(), "conclusion_marker"))
    matches = explicit or conclusions
    if matches:
        _, candidate, method = max(matches, key=lambda item: item[0])
        return {"candidate": candidate or None, "components": [], "extraction_method": method, "box_spans": []}

    words = re.findall(r"\S+", post_think)
    sentence_ends = len(re.findall(r"[.!?](?:\s|$)", post_think))
    if post_think and len(post_think) <= 256 and len(words) <= 8 and sentence_ends <= 1:
        return {"candidate": post_think.strip(), "components": [], "extraction_method": "concise_region", "box_spans": []}
    return {"candidate": None, "components": [], "extraction_method": "none", "box_spans": []}


def normalize_text_alias(value: str) -> str | None:
    text = value.strip().lower().strip("$ .,:;!")
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
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
        "\\emptyset",
        "\\varnothing",
        "∅",
        "{}",
    }
    return "empty_solution_set" if text in aliases else None


def math_variants(candidate: str, *, boxed_assignment: bool) -> list[str]:
    variants = [candidate, f"${candidate}$"]
    if boxed_assignment and "=" in candidate:
        right_hand_side = candidate.split("=", 1)[1].strip()
        if right_hand_side:
            variants.extend([right_hand_side, f"${right_hand_side}$"])
    result: list[str] = []
    for variant in variants:
        if variant not in result:
            result.append(variant)
    return result


def verify_single(candidate: str, correct: str, *, boxed_assignment: bool = False) -> dict[str, Any]:
    official, official_error = bounded_call(
        lambda: check_one_latex_answer(candidate, correct, extract_policy="flex", eval_policy="aggressive")
    )
    if isinstance(official, dict) and bool(official.get("is_correct")):
        return {
            "is_correct": True,
            "verifier_method": "official_checker",
            "official_checker_output": json_safe(official),
            "official_checker_error": official_error,
            "math_verify_result": [],
            "math_verify_error": [],
        }

    candidate_alias = normalize_text_alias(candidate)
    correct_alias = normalize_text_alias(correct)
    if candidate_alias is not None and candidate_alias == correct_alias:
        return {
            "is_correct": True,
            "verifier_method": "text_alias",
            "official_checker_output": json_safe(official),
            "official_checker_error": official_error,
            "math_verify_result": [],
            "math_verify_error": [],
        }

    attempts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for variant in math_variants(candidate, boxed_assignment=boxed_assignment):
        def run_math_verify() -> bool:
            from math_verify import parse, verify

            gold = parse(correct, parsing_timeout=10, raise_on_error=True)
            target = parse(variant, parsing_timeout=10, raise_on_error=True)
            return bool(verify(gold, target, timeout_seconds=10, raise_on_error=True))

        result, error = bounded_call(run_math_verify)
        attempts.append({"candidate": variant, "result": bool(result) if error is None else False})
        if error is not None:
            errors.append({"candidate": variant, **error})
        if result is True:
            return {
                "is_correct": True,
                "verifier_method": "math_verify",
                "official_checker_output": json_safe(official),
                "official_checker_error": official_error,
                "math_verify_result": attempts,
                "math_verify_error": errors,
            }
    return {
        "is_correct": False,
        "verifier_method": "failed",
        "official_checker_output": json_safe(official),
        "official_checker_error": official_error,
        "math_verify_result": attempts,
        "math_verify_error": errors,
    }


def split_reference_collection(correct: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"\s*(?:,|;|\band\b)\s*", correct, flags=re.IGNORECASE) if piece.strip()]


def verify_collection(components: list[str], correct: str) -> dict[str, Any]:
    references = split_reference_collection(correct)
    audit: list[dict[str, Any]] = []
    if len(references) != len(components):
        return {
            "is_correct": False,
            "verifier_method": "multi_component_count_mismatch",
            "official_checker_output": audit,
            "official_checker_error": None,
            "math_verify_result": [],
            "math_verify_error": [],
        }
    cache: dict[tuple[int, int], dict[str, Any]] = {}

    def pair_result(component_index: int, reference_index: int) -> dict[str, Any]:
        key = (component_index, reference_index)
        if key not in cache:
            cache[key] = verify_single(components[component_index], references[reference_index])
            audit.append(
                {
                    "component_index": component_index,
                    "reference_index": reference_index,
                    "component": components[component_index],
                    "reference": references[reference_index],
                    "verification": cache[key],
                }
            )
        return cache[key]

    def match(component_index: int, used: set[int]) -> bool:
        if component_index == len(components):
            return True
        for reference_index in range(len(references)):
            if reference_index in used:
                continue
            if pair_result(component_index, reference_index)["is_correct"] and match(
                component_index + 1, used | {reference_index}
            ):
                return True
        return False

    correct_match = match(0, set())
    return {
        "is_correct": correct_match,
        "verifier_method": "multi_symbolic_match" if correct_match else "multi_symbolic_failed",
        "official_checker_output": json_safe(audit),
        "official_checker_error": None,
        "math_verify_result": [],
        "math_verify_error": [],
    }


def score_answer(generated_text: str | None, correct: str) -> dict[str, Any]:
    post_think, think_status = isolate_final_region(generated_text)
    extraction = extract_candidate(post_think)
    candidate = extraction["candidate"]
    base = {
        "candidate": candidate,
        "extraction_method": extraction["extraction_method"],
        "box_spans": extraction["box_spans"],
        "box_contents": [box["content"] for box in extraction["box_spans"]],
        "think_status": think_status,
        "parse_failure": candidate is None,
        "scorer_version": SCORER_VERSION,
        "post_think_sha256": hashlib.sha256(post_think.encode("utf-8")).hexdigest(),
    }
    if candidate is None:
        return {
            **base,
            "is_correct": False,
            "verifier_method": "not_run",
            "official_checker_output": None,
            "official_checker_error": None,
            "math_verify_result": [],
            "math_verify_error": [],
        }
    components = extraction["components"]
    if len(components) > 1:
        verification = verify_collection(components, correct)
    else:
        verification = verify_single(
            candidate,
            correct,
            boxed_assignment=extraction["extraction_method"] in {"boxed", "fbox"} and "=" in candidate,
        )
    return {**base, **verification}


def metrics_from_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(bool(row["is_correct"]) for row in predictions)
    cap_rows = [row for row in predictions if bool(row["cap_hit"])]
    non_cap_rows = [row for row in predictions if not bool(row["cap_hit"])]
    return {
        "scorer_version": SCORER_VERSION,
        "num_eval_problems": total,
        "correct_count": correct,
        "accuracy": correct / total if total else 0.0,
        "cap_hit_count": len(cap_rows),
        "cap_hit_rate": len(cap_rows) / total if total else 0.0,
        "cap_hit_accuracy": sum(bool(row["is_correct"]) for row in cap_rows) / len(cap_rows) if cap_rows else 0.0,
        "non_cap_accuracy": (
            sum(bool(row["is_correct"]) for row in non_cap_rows) / len(non_cap_rows) if non_cap_rows else 0.0
        ),
        "parse_failures": sum(bool(row["parse_failure"]) for row in predictions),
        "average_generated_length": (
            sum(int(row["generated_token_count"]) for row in predictions) / total if total else 0.0
        ),
        "extraction_method_counts": dict(sorted(Counter(row["extraction_method"] for row in predictions).items())),
        "verifier_method_counts": dict(sorted(Counter(row["verifier_method"] for row in predictions).items())),
    }
