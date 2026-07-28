from __future__ import annotations

import pytest

from examples.qwen3_8b_32b_full_repro.math500_scorer_v4 import (
    SCORER_VERSION,
    score_response,
)


def score(text: str, gold: str):
    return score_response(
        text,
        gold,
        finish_reason="stop",
        stop_token_id=151645,
        cap_hit=False,
        problem_id="units",
    )


def test_v4_version_is_distinct_from_immutable_v3() -> None:
    from examples.qwen3_8b_32b_full_repro import math500_scorer

    assert math500_scorer.SCORER_VERSION == "math500_strict_boxed_scorer_v3"
    assert SCORER_VERSION == "math500_strict_boxed_units_scorer_v4"


@pytest.mark.parametrize(
    ("text", "gold"),
    [
        (r"\boxed{12 \, \text{cm}}", "12"),
        (r"\boxed{15}", r"15\mbox{ cm}^2"),
        (r"\boxed{\frac{16}{27}\text{ gallons}}", r"\frac{16}{27}"),
        (r"\boxed{145}", r"145^\circ"),
        (r"\boxed{\$36}", "36"),
    ],
)
def test_explicit_answer_units_are_normalized(
    text: str, gold: str
) -> None:
    result = score(text, gold)
    assert result["is_correct"], result
    normalization = result["typed_verifier_evidence"]["unit_normalization"]
    assert (
        normalization["candidate"]["units_removed"]
        or normalization["gold"]["units_removed"]
    )


def test_categorical_text_is_not_removed() -> None:
    result = score(r"\boxed{\text{Evelyn}}", r"\text{Evelyn}")
    assert result["is_correct"]
    assert not result["typed_verifier_evidence"]["unit_normalization"][
        "candidate"
    ]["units_removed"]


def test_normalizing_degrees_does_not_accept_the_wrong_value() -> None:
    assert not score(r"\boxed{125^\circ}", r"145^\circ")["is_correct"]


def test_cap_hits_remain_officially_incorrect() -> None:
    result = score_response(
        r"\boxed{12 \, \text{cm}}",
        "12",
        finish_reason="length",
        stop_token_id=None,
        cap_hit=True,
        problem_id="capped-units",
    )
    assert not result["is_correct"]
    assert result["official_decision_reason"] == "cap_hit"
    assert result["cap_counterfactual_diagnostic"]["is_correct"]
