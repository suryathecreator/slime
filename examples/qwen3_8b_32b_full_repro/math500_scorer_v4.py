#!/usr/bin/env python3
"""Strict boxed-answer MATH-500 scorer with explicit unit normalization."""

from __future__ import annotations

import re
from typing import Any, Callable

from examples.qwen3_8b_32b_full_repro import math500_scorer as v3


SCORER_VERSION = "math500_strict_boxed_units_scorer_v4"
TRACE_SCHEMA_VERSION = "math500_strict_boxed_units_trace_v4"

LATEX_UNIT_SUFFIX_RE = re.compile(
    r"""(?is)
    (?P<spacing>(?:\s|\\[,;:!]|\\quad|\\qquad)*)
    \\(?:text|mathrm|mbox)\s*\{(?P<unit>[^{}]+)\}
    (?P<power>\s*\^\s*(?:\{?\s*[23]\s*\}?))?
    \s*$
    """,
    re.VERBOSE,
)
PLAIN_UNIT_SUFFIX_RE = re.compile(
    r"""(?is)
    (?P<spacing>(?:\s|\\[,;:!]|\\quad|\\qquad)+)
    (?P<unit>
        (?:(?:square|sq\.?|cubic)\s+)?
        (?:millimet(?:er|re)s?|centimet(?:er|re)s?|kilomet(?:er|re)s?|
           meters?|metres?|inches?|feet|foot|yards?|miles?|
           grams?|kilograms?|ounces?|pounds?|
           millilit(?:er|re)s?|lit(?:er|re)s?|gallons?|
           seconds?|minutes?|hours?|days?|weeks?|years?|
           degrees?|radians?|dollars?|cents?|units?|
           mm|cm|dm|km|kg|mg|ml|mph|kph)
        (?:\s+(?:per|/)\s+
           (?:seconds?|minutes?|hours?|days?|meters?|metres?|
              kilomet(?:er|re)s?|feet|miles?|s|h))?
    )
    (?P<power>\s*\^\s*(?:\{?\s*[23]\s*\}?))?
    \s*$
    """,
    re.VERBOSE,
)
DEGREE_SUFFIX_RE = re.compile(
    r"(?is)(?:\s|\\[,;:!]|\\quad|\\qquad)*\^\s*\{?\s*\\circ\s*\}?\s*$"
)
CURRENCY_PREFIX_RE = re.compile(r"^\s*\\\$\s*")
UNIT_PHRASE_RE = re.compile(
    r"""(?is)^
    (?:(?:square|sq\.?|cubic)\s+)?
    (?:millimet(?:er|re)s?|centimet(?:er|re)s?|kilomet(?:er|re)s?|
       meters?|metres?|inches?|feet|foot|yards?|miles?|
       grams?|kilograms?|ounces?|pounds?|
       millilit(?:er|re)s?|lit(?:er|re)s?|gallons?|
       seconds?|minutes?|hours?|days?|weeks?|years?|
       degrees?|radians?|dollars?|cents?|units?|
       mm|cm|dm|km|kg|mg|ml|mph|kph)
    (?:\s+(?:per|/)\s+
       (?:seconds?|minutes?|hours?|days?|meters?|metres?|
          kilomet(?:er|re)s?|feet|miles?|s|h))?
    $""",
    re.VERBOSE,
)


def normalize_answer_units(value: str) -> dict[str, Any]:
    """Remove only recognized explicit presentation units from an answer."""

    original = str(value)
    normalized = original.strip()
    removed: list[str] = []
    currency = CURRENCY_PREFIX_RE.match(normalized)
    if currency:
        removed.append(currency.group(0).strip())
        normalized = normalized[currency.end() :].strip()
    degree = DEGREE_SUFFIX_RE.search(normalized)
    if degree:
        removed.append(degree.group(0).strip())
        normalized = normalized[: degree.start()].rstrip()
    latex = LATEX_UNIT_SUFFIX_RE.search(normalized)
    if latex:
        phrase = re.sub(r"\s+", " ", latex.group("unit").strip())
        if UNIT_PHRASE_RE.fullmatch(phrase):
            removed.append(
                latex.group(0)[len(latex.group("spacing")) :].strip()
            )
            normalized = normalized[: latex.start()].rstrip()
    else:
        plain = PLAIN_UNIT_SUFFIX_RE.search(normalized)
        if plain:
            removed.append(
                plain.group(0)[len(plain.group("spacing")) :].strip()
            )
            normalized = normalized[: plain.start()].rstrip()
    return {
        "original": original,
        "normalized": normalized,
        "removed_units": removed,
        "units_removed": bool(removed),
    }


def _normalization(candidate: str, gold: str) -> dict[str, Any]:
    return {
        "candidate": normalize_answer_units(candidate),
        "gold": normalize_answer_units(gold),
    }


def verify_equivalence(
    candidate: str,
    gold: str,
    timeout_seconds: int = 15,
    *,
    verifier: Callable[[str, str], dict[str, Any]] | None = None,
    force_scalar: bool = False,
) -> dict[str, Any]:
    normalization = _normalization(candidate, gold)
    result = v3.verify_equivalence(
        normalization["candidate"]["normalized"],
        normalization["gold"]["normalized"],
        timeout_seconds=timeout_seconds,
        verifier=verifier,
        force_scalar=force_scalar,
    )
    result["unit_normalization"] = normalization
    return result


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
    trace = v3.score_response(
        generated_text,
        gold_answer,
        finish_reason=finish_reason,
        stop_token_id=stop_token_id,
        cap_hit=cap_hit,
        problem_id=problem_id,
        parse_fn=parse_fn,
        verifier=lambda _candidate, _gold: {
            "parse_success": True,
            "gold_parsed": None,
            "candidate_parsed": None,
            "equivalent": False,
            "exception": None,
            "timeout": False,
        },
    )
    selected = trace.get("boxed_answer_candidate")
    normalization = _normalization(str(selected or ""), str(gold_answer))
    verification = trace["typed_verifier_evidence"]
    if selected is not None:
        verification = verify_equivalence(
            str(selected),
            str(gold_answer),
            verifier=verifier,
        )
        is_scorable = bool(verification.get("parse_success"))
        is_correct = bool(is_scorable and verification.get("equivalent"))
        if not is_scorable:
            decision_reason = "unscorable_verifier_failure"
        elif is_correct:
            decision_reason = "correct"
        else:
            decision_reason = "incorrect_math"
        trace["parse_result"] = verification.get("candidate_parsed")
        trace["parser_verifier_exception"] = verification.get("exception")
        trace["parser_verifier_timeout"] = bool(
            verification.get("timeout")
        )
        trace["math_verify_result"] = False if cap_hit else is_correct
        trace["is_scorable"] = False if cap_hit else is_scorable
        trace["is_correct"] = False if cap_hit else is_correct
        trace["official_decision_reason"] = (
            "cap_hit" if cap_hit else decision_reason
        )
        if cap_hit:
            counterfactual = trace["cap_counterfactual_diagnostic"]
            counterfactual.update(
                {
                    "parseable": is_scorable,
                    "parse_result": verification.get("candidate_parsed"),
                    "is_scorable": is_scorable,
                    "is_correct": is_correct,
                    "decision_reason": decision_reason,
                    "verification": verification,
                }
            )
    trace["scorer_version"] = SCORER_VERSION
    trace["trace_schema_version"] = TRACE_SCHEMA_VERSION
    trace["typed_verifier_evidence"] = verification
    trace["typed_verifier_evidence"]["unit_normalization"] = normalization
    counterfactual = trace.get("cap_counterfactual_diagnostic")
    if counterfactual is not None:
        counterfactual["verification"]["unit_normalization"] = normalization
    return trace


def main() -> None:
    import argparse
    import json

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
