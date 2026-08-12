from __future__ import annotations

import pytest

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.aime_scorer import (
    SCORER_VERSION,
    parse_aime_integer,
    score_response,
)

NUM_GPUS = 0


def score(text: str, gold: int = 42, *, cap_hit: bool = False):
    return score_response(
        text,
        gold,
        finish_reason="length" if cap_hit else "stop",
        stop_token_id=None if cap_hit else 151645,
        cap_hit=cap_hit,
        problem_id="test",
    )


def test_version_is_explicit() -> None:
    assert SCORER_VERSION == "aime_last_boxed_integer_scorer_v1"


def test_whole_response_last_valid_box_wins() -> None:
    result = score(r"<think>guess \boxed{7}</think> final \boxed{042}")
    assert result["is_correct"]
    assert result["parsed_integer"] == 42
    assert result["selected_box_event_id"] == "box_0001"


def test_empty_or_unclosed_later_box_does_not_replace_last_valid() -> None:
    assert score(r"\boxed{42} later \boxed{}")["is_correct"]
    result = score(r"\boxed{42} later \boxed{")
    assert result["is_correct"]
    assert result["malformed_boxes"][0]["reason"] == "unclosed_box"


def test_nested_and_escaped_braces_are_balanced() -> None:
    result = score(r"noise \boxed{\text{bad \} brace}} and \boxed{042}")
    assert result["is_correct"]
    assert result["official_candidate_raw"] == "042"
    assert len(result["box_events"]) == 2


@pytest.mark.parametrize(
    "candidate",
    ["42", "+42", "00042", r"\text{42}", r"\mathrm{042}", "$42$", r"\(42\)"],
)
def test_fixed_numeric_presentations(candidate: str) -> None:
    assert parse_aime_integer(candidate) == 42


@pytest.mark.parametrize(
    "candidate",
    ["-42", "42.", "42 answer", "6*7", r"\frac{84}{2}", "1000", ""],
)
def test_nonexact_or_out_of_range_values_are_rejected(candidate: str) -> None:
    assert parse_aime_integer(candidate) is None


def test_no_multipart_unboxed_or_prose_fallback() -> None:
    assert not score("Therefore the answer is 42.")["is_correct"]
    assert not score(r"\boxed{4}, \boxed{2}")["is_correct"]
    assert not score(r"\boxed{answer=42}")["is_correct"]


def test_cap_hit_does_not_change_the_official_score() -> None:
    natural = score(r"\boxed{42}")
    capped = score(r"\boxed{42}", cap_hit=True)
    assert natural["is_correct"] and capped["is_correct"]
    assert capped["cap_status"]


def test_invalid_gold_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid AIME gold"):
        score_response(
            r"\boxed{42}",
            "1000",
            finish_reason="stop",
            stop_token_id=151645,
            cap_hit=False,
            problem_id="bad",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
