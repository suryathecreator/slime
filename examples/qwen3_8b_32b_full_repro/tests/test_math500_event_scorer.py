from __future__ import annotations

import pytest

from examples.qwen3_8b_32b_full_repro.math500_scorer import score_response, verify_equivalence


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


def test_correct_candidate_followed_by_doubt_returns_no_answer() -> None:
    result = score("The answer is 42. Maybe this is wrong.")
    assert result["official_extracted_candidate"] is None
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


def test_maybe_is_not_equivalent_to_assertion() -> None:
    assert score("Maybe 42")["official_extracted_candidate"] is None
    assert score("The answer is 42")["official_extracted_candidate"] == "42"


def test_capped_generation_can_use_earlier_confident_answer() -> None:
    result = score("reasoning\nThe answer is 42.\nmore unfinished reasoning", cap=True)
    assert result["official_extracted_candidate"] == "42"
    assert result["cap_status"]


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
