from __future__ import annotations

import pytest

pytest.importorskip("math_verify")

from examples.qwen3_8b_32b_full_repro.math500_scorer_v5 import (
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
    assert SCORER_VERSION == "math500_last_boxed_symmetric_scorer_v5"


def test_last_complete_box_preserves_interior_verbatim() -> None:
    result = score(
        r"first \boxed{1} then \boxed{  \frac{2}{4}  }",
        r"\frac{1}{2}",
    )
    assert result["is_correct"]
    assert result["official_extracted_components_raw"] == [
        r"  \frac{2}{4}  "
    ]


def test_later_unclosed_or_empty_box_does_not_replace_last_valid_box() -> None:
    assert score(r"\boxed{7} later \boxed{", "7")["is_correct"]
    assert score(r"\boxed{7} later \boxed{}", "7")["is_correct"]


def test_nested_and_escaped_braces_are_balanced() -> None:
    result = score(r"\boxed{\frac{1}{\{2\}}}", r"\frac{1}{\{2\}}")
    assert result["official_extracted_components_raw"] == [
        r"\frac{1}{\{2\}}"
    ]


def test_adjacent_boxes_form_one_unordered_answer() -> None:
    result = score(r"\boxed{3}, \boxed{5} and \boxed{7}", "7,3,5")
    assert result["is_correct"]
    assert result["extraction_rule"] == "last_adjacent_boxed_group"


def test_prose_or_display_boundary_breaks_box_group() -> None:
    assert score(r"\boxed{3} therefore \boxed{5}", "5")["is_correct"]
    assert score(r"\[\boxed{3}\]\[\boxed{5}\]", "5")["is_correct"]


def test_unboxed_answer_is_always_wrong() -> None:
    result = score("Therefore the answer is 42.", "42")
    assert not result["is_correct"]
    assert result["official_decision_reason"] == "no_valid_complete_box"


def test_cap_hit_uses_the_same_score_as_a_natural_stop() -> None:
    natural = score(r"\boxed{42}", "42")
    capped = score(r"\boxed{42}", "42", cap_hit=True)
    assert natural["is_correct"] and capped["is_correct"]
    assert capped["cap_status"]


@pytest.mark.parametrize(
    ("candidate", "gold"),
    [
        (r"\frac{2}{4}", r"\frac{1}{2}"),
        (r"\sqrt{8}", r"2\sqrt{2}"),
        (r"p-q", r"p - q"),
        (r"x=2", r"x-2=0"),
        (r"x \in [-2,7]", r"x \in [-2,7]"),
        (r"(3,\frac{\pi}{2})", r"\left(3,\frac{\pi}{2}\right)"),
        (r"\{1,2\}", r"\{2,1\}"),
        (r"1\pm\sqrt{2}", r"1+\sqrt{2},1-\sqrt{2}"),
        (r"(2,12)\cup(12,102)", r"(12,102)\cup(2,12)"),
        (
            r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
            r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
        ),
    ],
)
def test_math500_equivalence_regressions(candidate: str, gold: str) -> None:
    assert verify_symmetric((candidate,), gold)["equivalent"]


def test_relations_do_not_match_only_their_rhs() -> None:
    assert not verify_symmetric(("x=2",), "2")["equivalent"]
    assert not verify_symmetric(("x=2",), "y=2")["equivalent"]


def test_ordered_tuple_requires_order_and_arity() -> None:
    assert not verify_symmetric(("(1,2)",), "(2,1)")["equivalent"]
    assert not verify_symmetric(("(1,2,3)",), "(1,2)")["equivalent"]


def test_finite_set_and_multi_answer_require_exact_cardinality() -> None:
    assert not verify_symmetric((r"\{1,2\}",), r"\{1,2,3\}")[
        "equivalent"
    ]
    assert not verify_symmetric(("1,2",), "1,2,3")["equivalent"]


def test_plus_minus_expands_inside_a_finite_set() -> None:
    gold = r"\{1\pm\sqrt{5},-2\}"
    candidate = r"\{-2,1-\sqrt{5},1+\sqrt{5}\}"
    assert verify_symmetric((candidate,), gold)["equivalent"]


def test_interval_requires_boundaries_and_endpoints() -> None:
    assert not verify_symmetric(("[0,1)",), "[0,1]")["equivalent"]
    assert not verify_symmetric(("[0,2)",), "[0,1)")["equivalent"]


def test_two_parentheses_emit_tuple_and_interval_without_gold_routing() -> None:
    kinds = {item.kind for item in interpret_answer("(1,2)")}
    assert kinds == {"ordered_tuple", "interval"}


def test_thousands_separator_does_not_consume_interval_comma() -> None:
    assert [item.kind for item in interpret_answer("58,500")] == [
        "expression"
    ]
    assert {item.kind for item in interpret_answer("(12,102)")} == {
        "ordered_tuple",
        "interval",
    }
    assert verify_symmetric(("58500",), "58,500")["equivalent"]


@pytest.mark.parametrize(
    ("candidate", "gold"),
    [
        (r"12\,\text{cm}", "12"),
        (r"145^\circ", "145"),
        (r"\$36", "36"),
        (r"15", r"15\mbox{ cm}^2"),
    ],
)
def test_recognized_presentation_units_are_symmetric(
    candidate: str, gold: str
) -> None:
    assert verify_symmetric((candidate,), gold)["equivalent"]


def test_categorical_text_is_not_removed_as_a_unit() -> None:
    assert verify_symmetric((r"\text{Evelyn}",), "evelyn")["equivalent"]
    assert not verify_symmetric((r"\text{Evelyn}",), "east")["equivalent"]


@pytest.mark.parametrize("candidate", ["C", "(C)", r"\text{(C)}"])
def test_multiple_choice_forms_are_equivalent(candidate: str) -> None:
    assert verify_symmetric((candidate,), r"\text{(C)}")["equivalent"]


def test_matrix_shape_is_guarded() -> None:
    candidate = r"\begin{pmatrix}1&2\\3&4\end{pmatrix}"
    gold = r"\begin{pmatrix}1&2&3\\4&0&0\end{pmatrix}"
    assert not verify_symmetric((candidate,), gold)["equivalent"]
