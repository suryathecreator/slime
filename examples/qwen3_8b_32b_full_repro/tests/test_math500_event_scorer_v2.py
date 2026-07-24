from __future__ import annotations

import pytest

from examples.qwen3_8b_32b_full_repro.math500_scorer_v2 import (
    SCORER_VERSION,
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


@pytest.mark.parametrize(
    ("text", "candidate"),
    [
        (r"</think>\boxed{42}", "42"),
        ("</think>The final answer is 42.", "42"),
        ("work\nTherefore, 42", "42"),
        ("work\n42", "42"),
    ],
)
def test_boxed_unboxed_conclusion_and_standalone_finals(text: str, candidate: str) -> None:
    assert score(text)["official_extracted_candidate"] == candidate


def test_multiple_boxes_choose_last_confident_replacement() -> None:
    result = score(r"</think>\boxed{41} then \boxed{42}")
    assert result["official_extracted_candidate"] == "42"
    assert result["answer_replacement_count"] == 1


def test_doubt_does_not_erase_a_complete_answer() -> None:
    result = score("The answer is 42. Maybe this is wrong.")
    assert result["official_extracted_candidate"] == "42"
    assert result["invalidation_flags"]["any_doubt"]


def test_correct_candidate_followed_by_wrong_replacement() -> None:
    result = score("The answer is 42. Actually, the answer is 41.")
    assert result["official_extracted_candidate"] == "41"
    assert not result["is_correct"]


def test_wrong_candidate_followed_by_correct_replacement() -> None:
    result = score("The answer is 41. Wait, actually, the answer is 42.")
    assert result["official_extracted_candidate"] == "42"
    assert result["is_correct"]


def test_reaffirmation_remains_confident() -> None:
    result = score("The answer is 42. Checking confirms it: the answer is 42.")
    assert result["official_extracted_candidate"] == "42"
    assert any(item["action"] == "reaffirmation" for item in result["selection_history"])


def test_v1_is_preserved_and_v2_is_active() -> None:
    from examples.qwen3_8b_32b_full_repro import math500_scorer_v1

    assert SCORER_VERSION == "math500_event_scorer_v2"
    assert math500_scorer_v1.SCORER_VERSION == "math500_event_scorer_v1"


def test_maybe_is_not_equivalent_to_assertion() -> None:
    assert score("Maybe 42")["official_extracted_candidate"] is None
    assert score("The answer is 42")["official_extracted_candidate"] == "42"


def test_capped_generation_can_use_earlier_confident_answer() -> None:
    result = score("reasoning\nThe answer is 42.\nmore unfinished reasoning", cap=True)
    assert result["official_extracted_candidate"] == "42"
    assert result["cap_status"]


def test_cap_ignores_weak_incidental_terminal_number() -> None:
    result = score("Working through the cases gives 100. Let me see", cap=True)
    assert result["official_extracted_candidate"] is None
    assert not result["is_correct"]


def test_cap_retains_answer_through_checking_language() -> None:
    result = score("The answer is 42.\nLet me verify this another way.", cap=True)
    assert result["official_extracted_candidate"] == "42"


def test_cap_suppresses_truncated_scalar_repeat() -> None:
    result = score("The answer is 59.\nChecking again: the answer is 59.\nThe answer is 5", gold="59", cap=True)
    assert result["official_extracted_candidate"] == "59"
    assert result["partial_cap_suppression_count"] == 1
    assert result["is_correct"]


def test_cap_suppresses_truncated_multipart_repeat() -> None:
    result = score(
        r"\boxed{3}, \boxed{5}, \boxed{7}" "\n" r"Repeating the result: \boxed{3}",
        gold="3, 5, 7",
        cap=True,
    )
    assert result["official_extracted_candidate"] == "3, 5, 7"
    assert result["partial_cap_suppression_count"] == 1


def test_cap_ignores_hypothetical_box_after_real_final() -> None:
    result = score(
        "**Final Answer**\n"
        r"\boxed{42}"
        "\nSuppose the answer were 41; then the answer would be "
        r"\boxed{41}",
        cap=True,
    )
    assert result["official_extracted_candidate"] == "42"


def test_distant_subproblem_negation_does_not_retract_answer() -> None:
    result = score(
        "The answer is 42. This resolves the requested problem. "
        + "Further checking of an intermediate count follows. " * 4
        + "That subproblem count is not 42 as previously thought.",
        cap=True,
    )
    assert result["official_extracted_candidate"] == "42"


def test_capped_latest_retracted_answer_returns_no_answer() -> None:
    result = score("The answer is 42. Scratch that; it is wrong and I am unsure.", cap=True)
    assert result["official_extracted_candidate"] is None


def test_tied_not_candidate_is_retracted() -> None:
    result = score("The answer is 42. No, not 42.")
    assert result["official_extracted_candidate"] is None
    assert result["invalidation_flags"]["any_retraction"]


def test_closed_think_never_uses_answer_only_inside_think() -> None:
    result = score("<think>The answer is 42.</think>Still working")
    assert result["official_extracted_candidate"] is None
    assert result["answer_anywhere_diagnostic"] == "42"


def test_closed_think_uses_unboxed_final_afterward() -> None:
    result = score("<think>work</think>\nThe answer is 42.")
    assert result["official_extracted_candidate"] == "42"


def test_structural_delimiter_cannot_replace_box() -> None:
    result = score("\\boxed{42}\nThe final answer is \\]")
    assert result["official_extracted_candidate"] == "42"
    assert result["structural_candidate_rejection_count"] >= 1


def test_natural_stop_prose_does_not_replace_strong_box() -> None:
    result = score(r"\boxed{42}" "\nThis completes the requested calculation.")
    assert result["official_extracted_candidate"] == "42"


def test_concise_standalone_sentence_preserves_terminal_punctuation() -> None:
    result = score("So, Hillary has 7 nickels.", gold="So, Hillary has 7 nickels.")
    assert result["official_extracted_candidate"] == "So, Hillary has 7 nickels."


def test_adjacent_boxes_are_one_exact_cardinality_answer() -> None:
    result = score(r"\boxed{3}, \boxed{5}, \boxed{7}", gold="7, 3, 5")
    assert result["official_extracted_candidate"] == "3, 5, 7"
    assert result["extraction_rule"] == "boxed_group"
    assert result["is_correct"]


def test_collection_requires_exact_cardinality() -> None:
    assert not score(r"\boxed{3}, \boxed{5}", gold="3, 5, 7")["is_correct"]


def test_malformed_box_is_recorded_without_guessing() -> None:
    result = score(r"</think>\boxed{MALFORMED")
    assert result["official_extracted_candidate"] is None
    assert result["invalidation_flags"]["malformed_boxes"]


def test_parse_failure_and_empty_generation_return_no_answer() -> None:
    malformed = score("The answer is MALFORMED 42")
    assert malformed["official_extracted_candidate"] is None
    assert malformed["parser_verifier_exception"] == "unparseable_candidate"
    assert score("")["official_extracted_candidate"] is None


def test_verifier_timeout_is_preserved() -> None:
    def timeout_verify(_candidate: str, _gold: str):
        return {"parse_success": False, "candidate_parsed": None, "equivalent": False, "exception": "timed out", "timeout": True}

    result = score_response(
        "The answer is 42",
        "42",
        finish_reason="stop",
        stop_token_id=151645,
        cap_hit=False,
        problem_id="timeout",
        parse_fn=fake_parse,
        verifier=timeout_verify,
    )
    assert not result["is_correct"]
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
    ],
)
def test_math_verify_equivalence_regressions(candidate: str, gold: str) -> None:
    pytest.importorskip("math_verify")
    result = verify_equivalence(candidate, gold)
    assert result["parse_success"], result
    assert result["equivalent"], result
