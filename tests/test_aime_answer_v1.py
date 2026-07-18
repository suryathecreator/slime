from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "examples" / "qwen3_1_7b_opd_aime2026" / "aime_answer_v1.py"
SPEC = importlib.util.spec_from_file_location("aime_answer_v1", MODULE_PATH)
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def score(text: str, *, gold: int = 42, cap_hit: bool = False):
    return SCORER.score_answer(text, gold, "problem", "aime_2026_01", cap_hit, "length" if cap_hit else "stop")


def test_natural_stop_uses_final_post_think_box():
    result = score("<think>\\boxed{7}</think> checking\n\\boxed{042}")
    assert result["is_correct"]
    assert result["parsed_answer"] == 42
    assert result["region_policy"] == "final_post_think_region"


def test_chronologically_last_strong_event_wins():
    result = score("</think>Answer is 17. Correction: final answer: 42")
    assert result["is_correct"]
    assert result["extraction_method"] == "explicit_answer_marker"


def test_standalone_unboxed_final_integer_is_allowed():
    result = score("</think>We finish the argument.\n42")
    assert result["is_correct"]
    assert result["extraction_method"] == "unboxed_final_numeric"


def test_incidental_last_number_in_calculation_is_rejected():
    result = score("</think>We still need to calculate x = 6 * 7")
    assert not result["is_correct"]
    assert result["decision_reason"] == "no_eligible_answer_event"


def test_cap_hit_accepts_latest_explicit_answer():
    result = score("Try 11. Answer is 42. Now I will verify this by considering 17", cap_hit=True)
    assert result["is_correct"]


def test_cap_hit_retains_answer_through_later_uncertainty():
    result = score("Answer is 42. I am not entirely sure and should check 17 cases", cap_hit=True)
    assert result["is_correct"]


def test_later_distinct_answer_event_replaces_prior_one():
    result = score("Answer is 11.\nAfter correction, final answer: 42", cap_hit=True)
    assert result["is_correct"]
    assert len(result["answer_events"]) == 2


def test_marker_does_not_reach_into_later_reasoning_for_number():
    result = score("</think>The answer is not settled. We should inspect 42 cases.")
    assert not result["is_correct"]


def test_cap_hit_rejects_random_terminal_calculation_number():
    result = score("We have x=6 and next compute y = x * 7", cap_hit=True)
    assert not result["is_correct"]


def test_noninteger_candidate_fails_parse():
    result = score("</think>\\boxed{41/1}", gold=41)
    assert not result["is_correct"]
    assert result["decision_reason"] == "integer_parse_failure"


def test_prethink_answer_is_ignored_after_natural_stop():
    result = score("<think>Answer is 42.</think>I could not finish the response.")
    assert not result["is_correct"]


def test_metrics_are_monte_carlo_single_sample_accuracy():
    rows = [score("</think>\\boxed{42}"), score("</think>\\boxed{9}")]
    metrics = SCORER.metrics_from_predictions(rows)
    assert metrics["pass_at_1_mc"] == 0.5
    assert metrics["total_correct"] == 1
