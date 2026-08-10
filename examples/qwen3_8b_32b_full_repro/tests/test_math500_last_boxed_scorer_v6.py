from __future__ import annotations

import pytest

pytest.importorskip("math_verify")

from examples.qwen3_8b_32b_full_repro.math500_scorer_v6 import (
    SCORER_VERSION,
    interpret_answer,
    score_response,
    verify_symmetric,
)


def score(text: str, gold: str, *, cap_hit: bool = False):
    return score_response(
        text,
        gold,
        finish_reason="length" if cap_hit else "stop",
        stop_token_id=None if cap_hit else 151645,
        cap_hit=cap_hit,
        problem_id="regression",
    )


def test_version_is_new_and_immutable() -> None:
    assert SCORER_VERSION == "math500_last_boxed_symmetric_scorer_v6"


def test_last_complete_box_preserves_interior_verbatim() -> None:
    result = score(
        r"first \boxed{1} then \boxed{  \frac{2}{4}  }",
        r"\frac{1}{2}",
    )
    assert result["is_correct"]
    assert result["official_extracted_components_raw"] == [
        r"  \frac{2}{4}  "
    ]


def test_unboxed_answer_is_always_wrong() -> None:
    result = score("Therefore the answer is 42.", "42")
    assert not result["is_correct"]
    assert result["official_decision_reason"] == "no_valid_complete_box"


def test_cap_hit_uses_the_same_score_as_a_natural_stop() -> None:
    assert score(r"\boxed{42}", "42")["is_correct"]
    capped = score(r"\boxed{42}", "42", cap_hit=True)
    assert capped["is_correct"] and capped["cap_status"]


@pytest.mark.parametrize(
    ("candidate", "gold"),
    [
        (r"\frac{2}{4}", r"\frac{1}{2}"),
        (r"\sqrt{8}", r"2\sqrt{2}"),
        (r"p-q", r"p - q"),
        (r"x=2", r"x-2=0"),
        (r"(3,\frac{\pi}{2})", r"\left(3,\frac{\pi}{2}\right)"),
        (r"\{1,2\}", r"\{2,1\}"),
        (r"1\pm\sqrt{2}", r"1+\sqrt{2},1-\sqrt{2}"),
        (r"(2,12)\cup(12,102)", r"(12,102)\cup(2,12)"),
        (r"10\%", "10"),
        ("0.5", r"\frac{1}{2}"),
    ],
)
def test_math500_equivalence_regressions(candidate: str, gold: str) -> None:
    assert verify_symmetric((candidate,), gold)["equivalent"]


@pytest.mark.parametrize(
    ("candidate", "gold"),
    [
        (r"-125x^{12}", "-125"),
        (r"1-i\sqrt{3}", "1"),
        (r"-9-12i", "1-12i"),
        (r"2x+2", "2k+2"),
        (r"2a+2b+3c+2k", "2k"),
        (r"1\frac{1}{2}\sqrt{5}", r"1\frac{4}{5}"),
        ("0.00006666666666666667", ".0000672"),
    ],
)
def test_partial_and_rounded_matches_are_rejected(
    candidate: str, gold: str
) -> None:
    assert not verify_symmetric((candidate,), gold)["equivalent"]


def test_relations_require_complete_relation_objects() -> None:
    assert not verify_symmetric(("x=2",), "2")["equivalent"]
    assert not verify_symmetric(("x=2",), "y=2")["equivalent"]


def test_structural_guards_remain_strict() -> None:
    assert not verify_symmetric(("(1,2)",), "(2,1)")["equivalent"]
    assert not verify_symmetric((r"\{1,2\}",), r"\{1,2,3\}")[
        "equivalent"
    ]
    assert not verify_symmetric(("[0,1)",), "[0,1]")["equivalent"]


def test_ambiguous_parentheses_remain_gold_independent() -> None:
    assert {item.kind for item in interpret_answer("(1,2)")} == {
        "ordered_tuple",
        "interval",
    }


def test_parser_evidence_records_the_complete_latex_parse() -> None:
    evidence = verify_symmetric((r"2x+2",), r"2x+2")
    attempt = evidence["attempts"][0]["evidence"]["attempts"][0]
    assert attempt["candidate_backend"] == "latex"
    assert attempt["candidate_parsed_repr"] == "2*x + 2"
