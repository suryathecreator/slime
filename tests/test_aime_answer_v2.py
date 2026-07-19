from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "examples" / "qwen3_1_7b_opd_aime2026" / "aime_answer_v2.py"
SPEC = importlib.util.spec_from_file_location("aime_answer_v2", MODULE_PATH)
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def score(text: str, *, gold: int = 42, cap_hit: bool = False):
    return SCORER.score_answer(text, gold, "problem", "aime_2026_01", cap_hit, "length" if cap_hit else "stop")


def test_post_think_region_is_used_for_natural_and_capped_outputs():
    natural = score(r"<think>\boxed{7}</think>\boxed{42}")
    capped = score(r"<think>\boxed{42}</think>\boxed{7}", cap_hit=True)
    assert natural["is_correct"]
    assert not capped["is_correct"]
    assert capped["candidate"] == "7"
    assert natural["region_policy"] == capped["region_policy"] == "final_post_think_region"


def test_last_closing_think_tag_defines_the_only_eligible_region():
    result = score(r"</think>\boxed{42}<think>draft</think>\boxed{9}")
    assert result["candidate"] == "9"
    assert not result["is_correct"]


def test_tagless_output_uses_last_chronological_answer():
    result = score("Answer: 11\nAfter correction, final answer: 42")
    assert result["is_correct"]
    assert result["replacement_count"] == 1


def test_conservative_standalone_numeric_can_replace_an_earlier_box():
    result = score("\\boxed{11}\n42")
    assert result["is_correct"]
    assert result["extraction_method"] == "unboxed_final_numeric"


def test_explicit_retraction_without_replacement_is_incorrect():
    result = score(r"</think>\boxed{42}. I was wrong.")
    assert not result["is_correct"]
    assert result["decision_reason"] == "explicit_retraction_without_replacement"
    assert result["retraction_count"] == 1


def test_answer_after_retraction_replaces_the_retracted_answer():
    result = score(r"</think>Answer: 11. I retract that answer. Final answer: 42")
    assert result["is_correct"]
    assert result["candidate"] == "42"
    assert any(item["action"] == "replacement_after_retraction" for item in result["selection_trace"])


def test_generic_uncertainty_does_not_retract_an_answer():
    result = score(r"</think>\boxed{42}. Actually, let me check the derivation.")
    assert result["is_correct"]
    assert result["retraction_count"] == 0


def test_prethink_retraction_does_not_affect_post_think_answer():
    result = score(r"<think>Answer: 42. I was wrong.</think>\boxed{42}")
    assert result["is_correct"]
    assert result["retraction_count"] == 0


def test_exact_arithmetic_equality_fixes_observed_false_negative():
    result = score(r"</think>\boxed{41 + 9 = 50}", gold=50)
    assert result["is_correct"]
    assert result["parsed_answer"] == 50


def test_marker_preserves_and_scores_an_arithmetic_expression():
    result = score(r"</think>Final answer: 6 \times (8 - 1) = 42 because the sum is exact.")
    assert result["is_correct"]
    assert result["candidate"] == r"6 \times (8 - 1) = 42"


def test_exact_division_and_terminal_punctuation_are_supported():
    assert score(r"</think>\boxed{84 / 2.}")["is_correct"]
    assert score(r"</think>\boxed{42.}")["is_correct"]


def test_inconsistent_equality_is_rejected_even_if_rhs_matches_gold():
    result = score(r"</think>\boxed{41 + 8 = 50}", gold=50)
    assert not result["is_correct"]
    assert result["decision_reason"] == "arithmetic_parse_failure"


def test_wrong_but_consistent_arithmetic_is_an_integer_mismatch():
    result = score(r"</think>\boxed{100 \cdot 1 + 81 = 181}", gold=754)
    assert not result["is_correct"]
    assert result["parsed_answer"] == 181
    assert result["decision_reason"] == "integer_mismatch"


def test_symbolic_roots_variables_powers_and_noninteger_values_are_rejected():
    for candidate in (r"\sqrt{42}", "x = 42", "6 ** 2 + 6", "85 / 2"):
        result = score(f"</think>\\boxed{{{candidate}}}")
        assert not result["is_correct"]
        assert result["parsed_answer"] is None


def test_incidental_last_calculation_number_is_not_an_answer():
    result = score("</think>We still need to calculate x = 6 * 7")
    assert not result["is_correct"]
    assert result["decision_reason"] == "no_eligible_answer_event"


def test_marker_does_not_reach_into_later_prose_for_a_number():
    result = score("</think>The answer is not settled. We should inspect 42 cases.")
    assert not result["is_correct"]
    assert result["candidate"] is None


def test_metrics_report_mc_accuracy_and_selection_audit():
    rows = [score("</think>\\boxed{42}"), score("</think>\\boxed{9}. I was wrong")]
    metrics = SCORER.metrics_from_predictions(rows)
    assert metrics["pass_at_1_mc"] == 0.5
    assert metrics["total_correct"] == 1
    assert metrics["retraction_count"] == 1
    assert metrics["region_policy_counts"] == {"final_post_think_region": 2}
