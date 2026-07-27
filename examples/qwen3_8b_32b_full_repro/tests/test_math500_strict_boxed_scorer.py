from __future__ import annotations

import pytest

from examples.qwen3_8b_32b_full_repro.math500_scorer import (
    SCORER_VERSION,
    classify_think_tags,
    score_response,
    verify_equivalence,
)


def fake_parse(value: str, **_kwargs):
    value = value.strip()
    if not value or "MALFORMED" in value:
        return []
    return [value]


def fake_verify(candidate: str, gold: str):
    normalize = lambda value: value.strip().strip("$ ").replace(" ", "")
    return {
        "parse_success": True,
        "gold_parsed": gold,
        "candidate_parsed": candidate,
        "equivalent": normalize(candidate) == normalize(gold),
        "exception": None,
        "timeout": False,
    }


def score(text: str, gold: str = "42", *, cap: bool = False):
    return score_response(
        text,
        gold,
        finish_reason="length" if cap else "stop",
        stop_token_id=None if cap else 151645,
        cap_hit=cap,
        problem_id="test",
        parse_fn=fake_parse,
        verifier=fake_verify,
    )


def test_versions_preserve_v1_v2_and_activate_v3() -> None:
    from examples.qwen3_8b_32b_full_repro import math500_scorer_v1, math500_scorer_v2

    assert math500_scorer_v1.SCORER_VERSION == "math500_event_scorer_v1"
    assert math500_scorer_v2.SCORER_VERSION == "math500_event_scorer_v2"
    assert SCORER_VERSION == "math500_strict_boxed_scorer_v3"


@pytest.mark.parametrize(
    ("text", "state"),
    [
        (r"\boxed{42}", "none"),
        (r"<think>work</think>\boxed{42}", "complete"),
        (r"<think>work \boxed{42}", "open_without_close"),
        (r"work</think>\boxed{42}", "close_without_open"),
        (r"<think>work</think></think>\boxed{42}", "malformed_or_repeated"),
        (r"</think>work<think>\boxed{42}", "malformed_or_repeated"),
    ],
)
def test_think_tag_states(text: str, state: str) -> None:
    assert classify_think_tags(text)["state"] == state
    assert score(text)["think_tag_state"] == state


def test_single_complete_think_still_uses_the_whole_response() -> None:
    result = score(r"<think>\boxed{41}</think>work \boxed{42}")
    assert result["official_extracted_candidate"] == "42"
    assert result["extraction_region"] == "whole_response_all_think_states"
    no_final = score(r"<think>\boxed{42}</think>still working")
    assert no_final["official_extracted_candidate"] == "42"
    assert no_final["is_correct"]


@pytest.mark.parametrize(
    "text",
    [
        r"<think>unfinished \boxed{42}",
        r"reasoning</think>\boxed{42}",
        r"<think>first</think></think>\boxed{42}",
        r"</think>first<think>\boxed{42}",
    ],
)
def test_every_think_state_uses_whole_response(text: str) -> None:
    result = score(text)
    assert result["official_extracted_candidate"] == "42"
    assert result["extraction_region"] == "whole_response_all_think_states"


def test_unboxed_answers_never_score() -> None:
    for text in ("The answer is 42.", "Therefore, 42", "work\n42", ""):
        result = score(text)
        assert result["official_extracted_candidate"] is None
        assert result["official_decision_reason"] == "unscorable_no_box"
        assert not result["is_correct"]


def test_last_complete_box_wins_without_discourse_analysis() -> None:
    result = score(
        r"\boxed{42} I retract that and keep reasoning. "
        r"Suppose a different answer were \boxed{41}."
    )
    assert result["official_extracted_candidate"] == "41"
    assert not result["is_correct"]
    assert result["selection_history"][0]["ignored_later_reasoning"]


def test_later_prose_does_not_change_last_box() -> None:
    result = score(r"\boxed{42} Actually this may be wrong; I will keep checking without another box.")
    assert result["official_extracted_candidate"] == "42"
    assert result["is_correct"]


def test_adjacent_boxes_are_one_multipart_answer() -> None:
    result = score(r"\boxed{3}, \boxed{5}, \boxed{7}", gold="7, 3, 5")
    assert result["official_extracted_candidate"] == "3, 5, 7"
    assert result["extraction_rule"] == "last_adjacent_boxed_group"
    assert result["is_correct"]


def test_separated_boxes_select_the_last_group() -> None:
    result = score("\\boxed{41}\nThen more work leads to\n\\boxed{42}")
    assert result["official_extracted_candidate"] == "42"
    assert len(result["boxed_groups"]) == 2


def test_nested_braces_are_preserved() -> None:
    result = score(r"\boxed{\frac{1}{2}}", gold=r"\frac{1}{2}")
    assert result["official_extracted_candidate"] == r"\frac{1}{2}"
    assert result["is_correct"]


def test_cap_hit_is_officially_wrong_but_keeps_strict_box_diagnostic() -> None:
    result = score(r"\boxed{42}", cap=True)
    assert result["official_extracted_candidate"] is None
    assert result["official_decision_reason"] == "cap_hit"
    assert not result["is_correct"]
    assert not result["is_scorable"]
    assert result["cap_counterfactual_diagnostic"]["candidate"] == "42"
    assert result["cap_counterfactual_diagnostic"]["is_correct"]


def test_cap_without_box_keeps_negative_counterfactual() -> None:
    result = score("The answer is 42", cap=True)
    assert result["official_decision_reason"] == "cap_hit"
    assert not result["cap_counterfactual_diagnostic"]["is_scorable"]
    assert not result["cap_counterfactual_diagnostic"]["is_correct"]


def test_selected_parse_failure_is_unscorable_without_earlier_fallback() -> None:
    result = score(r"\boxed{42} then \boxed{MALFORMED}")
    assert result["official_extracted_candidate"] == "MALFORMED"
    assert not result["is_scorable"]
    assert not result["is_correct"]
    assert result["official_decision_reason"] == "unscorable_parse_failure"


def test_empty_or_malformed_box_is_reported_unscorable() -> None:
    empty = score(r"\boxed{}")
    assert empty["official_decision_reason"] == "unscorable_malformed_or_empty_box"
    assert empty["structural_candidate_rejection_count"] == 1
    malformed = score(r"\boxed{42")
    assert malformed["official_decision_reason"] == "unscorable_malformed_or_empty_box"
    assert malformed["invalidation_flags"]["malformed_boxes"]


def test_verifier_failure_is_unscorable() -> None:
    def failed_verify(_candidate: str, _gold: str):
        return {
            "parse_success": False,
            "gold_parsed": None,
            "candidate_parsed": None,
            "equivalent": False,
            "exception": "timed out",
            "timeout": True,
        }

    result = score_response(
        r"\boxed{42}",
        "42",
        finish_reason="stop",
        stop_token_id=151645,
        cap_hit=False,
        problem_id="timeout",
        parse_fn=fake_parse,
        verifier=failed_verify,
    )
    assert result["official_decision_reason"] == "unscorable_verifier_failure"
    assert not result["is_scorable"]
    assert result["parser_verifier_timeout"]


@pytest.mark.parametrize(
    ("candidate", "gold"),
    [
        (r"\frac{1}{2}", r"\frac{2}{4}"),
        (r"\sqrt{8}", r"2\sqrt{2}"),
        (r"[0,1)", r"[0,1)"),
        (r"\{1,2\}", r"\{2,1\}"),
        (r"(1,2)", r"(1,2)"),
        (r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", r"\begin{pmatrix}1&2\\3&4\end{pmatrix}"),
        (r"x=2", r"x-2=0"),
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"1\pm\sqrt{2}", r"1+\sqrt{2}, 1-\sqrt{2}"),
    ],
)
def test_math500_typed_equivalence_regressions(candidate: str, gold: str) -> None:
    pytest.importorskip("math_verify")
    result = verify_equivalence(candidate, gold)
    assert result["parse_success"], result
    assert result["equivalent"], result


def test_ordered_tuple_is_not_reordered() -> None:
    pytest.importorskip("math_verify")
    result = verify_equivalence("(1,2)", "(2,1)")
    assert result["parse_success"], result
    assert not result["equivalent"], result
