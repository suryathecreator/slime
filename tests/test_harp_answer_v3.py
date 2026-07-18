from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/qwen3_4b_opd_harp"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def deliberately_aggressive_official(candidate, correct, **_kwargs):
    # Reproduce the class of upstream false positive that V3 must ignore.
    if "c_1" in str(candidate) and "sum c_k" in str(correct):
        return {"is_correct": True, "predict": candidate}
    letters = lambda value: "".join(re.findall(r"[A-Za-z]", str(value))).lower()
    return {"is_correct": letters(candidate) == letters(correct), "predict": candidate}


def fake_parse(value, **_kwargs):
    text = re.sub(r"\s+", "", str(value)).strip("$").replace(r"\ge", ">=").replace(r"\le", "<=")
    if text == "FAIL":
        return []
    return [text]


def fake_verify(gold, candidate, **_kwargs):
    left = gold[0]
    right = candidate[0]
    equivalent_pairs = {
        frozenset({"a>=3b", "b<=a/3"}),
        frozenset({r"\frac{1}{2}", "0.5"}),
    }
    return left == right or frozenset({left, right}) in equivalent_pairs


harp_package = types.ModuleType("harp_official")
harp_checker = types.ModuleType("harp_official.latex_answer_check")
harp_checker.check_one_latex_answer = deliberately_aggressive_official
math_verify = types.ModuleType("math_verify")
math_verify.parse = fake_parse
math_verify.verify = fake_verify
sys.modules.setdefault("harp_official", harp_package)
sys.modules.setdefault("harp_official.latex_answer_check", harp_checker)
sys.modules.setdefault("math_verify", math_verify)
sys.path.insert(0, str(EXAMPLE))

scorer = load_module("harp_answer_v3_test", EXAMPLE / "harp_answer_v3.py")


def test_v2_and_official_checker_are_byte_frozen():
    expected = {
        "harp_answer_v2.py": "76524e2e5a6659ccb354928ebc014b5c3fecf17bedff38c5e2610e0d659beae9",
        "harp_official/latex_answer_check.py": "54b191b9ec51be29538e881d2b0abd64cb176bdb822511a8d4a1694faedf1af0",
        "harp_official/parsing_lib.py": "5ca6f58beddabb87dbdbfb824cd8a4fc5c275087261df50725d27290ce42a9d0",
    }
    for relative, wanted in expected.items():
        assert hashlib.sha256((EXAMPLE / relative).read_bytes()).hexdigest() == wanted


def score(text: str, gold: str, *, cap_hit: bool = False, problem: str = "Find the answer."):
    return scorer.score_answer(text, gold, problem, "p1", cap_hit, "length" if cap_hit else "stop")


def test_natural_stop_uses_last_chronological_strong_event():
    result = score(r"<think>draft \boxed{9}</think>First \boxed{1}.\nFinal answer: 2", "2")
    assert result["candidate"] == "2"
    assert result["selected_event_id"] is not None
    assert result["grade"] == "correct"
    assert any(item["action"] == "replacement" for item in result["selection_trace"])


def test_adjacent_boxes_form_exact_multipart_answer():
    result = score(r"</think>\boxed{2}, \boxed{1}", "1 and 2")
    assert result["extraction_method"] == "adjacent_box_collection"
    assert result["components"] == ["2", "1"]
    assert result["verifier_evidence"]["gold_parse"]["cardinality"] == 2
    assert result["grade"] == "correct"


def test_natural_concise_fallback_but_no_arbitrary_last_number():
    assert score(r"</think>$5$", "5")["grade"] == "correct"
    long = "</think>This is a long calculation with many intermediate steps and it happens to end near 5"
    result = score(long, "5")
    assert result["candidate"] is None
    assert result["grade"] == "incorrect"


def test_unclosed_natural_thinking_has_no_final_region():
    result = score(r"<think>work \boxed{5}", "5")
    assert result["think_status"] == "unclosed"
    assert result["candidate"] is None


def test_cap_hit_keeps_answer_through_uncertainty_and_reaffirms():
    result = score("work\nAnswer: 5. But let me check.\nMore checking\nAnswer: 5", "5", cap_hit=True)
    assert result["candidate"] == "5"
    assert result["reaffirmation_count"] == 1
    assert result["grade"] == "correct"


def test_cap_hit_distinct_answer_replaces_and_unreplaced_retraction_fails():
    replaced = score("Answer: 5\nCorrection. Final answer: 7", "7", cap_hit=True)
    assert replaced["candidate"] == "7"
    assert replaced["replacement_count"] == 1
    assert replaced["grade"] == "correct"

    retracted = score("Answer: 5\nI was wrong", "5", cap_hit=True)
    assert retracted["grade"] == "incorrect"
    assert retracted["decision_reason"] == "explicit_retraction_without_replacement"


def test_cap_hit_ignores_generic_conclusions_and_incidental_numbers():
    result = score("We have x=3. Therefore 7. Then check 11.", "7", cap_hit=True)
    assert result["candidate"] is None
    assert result["grade"] == "incorrect"
    assert any(event["kind"] == "generic_conclusion" for event in result["answer_events"])


@pytest.mark.parametrize(
    ("value", "expected_kind"),
    [
        ("no solutions", "empty_solution"),
        (r"58\frac{1}{2}", "scalar"),
        ("48%", "percentage"),
        (r"2\sqrt{3}", "scalar"),
        ("$2.60 dollars", "quantity"),
        (r"x\ge 3", "inequality_or_equation"),
        ("[1, 3)", "interval"),
        ("11 to 5", "ratio"),
        ("{1, 2, 3}", "collection"),
        ("A", "point_label"),
        ("IV", "roman_label"),
        ("blue", "color_label"),
        (r"\infty", "infinity"),
        ("3 north, 4 east", "labeled_directions"),
    ],
)
def test_declared_answer_types_are_typed(value, expected_kind):
    parsed = scorer.parse_answer(value, "A ratio and a set may be requested.")
    assert parsed["success"] is True
    assert parsed["kind"] == expected_kind


def test_context_supported_dimensions_and_ratio_fraction():
    dimensions = scorer.parse_answer("4 by 8", "Find the dimensions of the rectangle.")
    assert dimensions["kind"] == "dimensions"
    assert scorer.parse_answer("11/5", "Find the ratio.", expected_kind="ratio")["kind"] == "ratio"
    assert score(r"</think>\boxed{\frac{11}{5}}", "11:5", problem="Find the ratio.")["grade"] == "correct"


def test_assignment_number_word_percentage_and_marker_tail():
    assert score(r"</think>\boxed{-\frac{1}{2}}", r"$m=-\frac{1}{2}$")["grade"] == "correct"
    assert score(r"</think>\boxed{2}", r"$\text{two}$")["grade"] == "correct"
    assert score(r"</think>\boxed{48}", r"$48\%$", problem="What percent?")["grade"] == "correct"
    tailed = score("Answer: 927, based on the previous calculations.", "927", cap_hit=True)
    assert tailed["candidate"] == "927"
    assert tailed["grade"] == "correct"


def test_compatible_units_and_conflicting_units_are_not_discarded():
    compatible = score(r"</think>\boxed{1\text{ foot}}", r"$12\text{ inches}$", problem="Give the length in inches.")
    assert compatible["grade"] == "correct"
    conflict = score(r"</think>\boxed{1\text{ minute}}", r"$1\text{ inch}$", problem="Give the length in inches.")
    assert conflict["grade"] == "incorrect"
    assert conflict["decision_reason"] == "conflicting_units"


def test_algebraically_equivalent_inequality_uses_successful_typed_parse():
    result = score(r"</think>\boxed{a\ge 3b}", r"$b\le a/3$")
    assert result["grade"] == "correct"
    assert result["verifier_evidence"]["typed_decision"]["math_verify"]["parse_success"] is True


def test_official_positive_cannot_override_typed_mismatch(monkeypatch):
    monkeypatch.setattr(
        scorer,
        "check_one_latex_answer",
        lambda candidate, _correct, **_kwargs: {"is_correct": True, "predict": candidate},
    )
    pi_mismatch = score(r"</think>\boxed{73\pi}", r"$65\pi$")
    assert pi_mismatch["verifier_evidence"]["official_checker"]["output"]["is_correct"] is True
    assert pi_mismatch["grade"] == "incorrect"

    unsupported_official = score(r"</think>\boxed{c_1}", r"$\sum c_k$")
    assert unsupported_official["verifier_evidence"]["official_checker"]["output"]["is_correct"] is True
    assert unsupported_official["grade"] == "incorrect"


def test_failed_parses_never_compare_equal_or_publish():
    left = scorer.parsed_failure("FAIL", "failed")
    right = scorer.parsed_failure("FAIL", "failed")
    assert scorer.compare_typed(left, right, "problem")["grade"] == "needs_review"
    row = {
        "grade": "needs_review",
        "cap_hit": False,
        "parse_failure": True,
        "generated_token_count": 1,
        "extraction_method": "explicit_final_marker",
        "decision_reason": "typed_symbolic_parse_failed",
        "verifier_method": "typed_scalar",
        "answer_events": [],
        "selection_trace": [],
    }
    with pytest.raises(ValueError, match="Refusing to publish"):
        scorer.metrics_from_predictions([row])


def test_collection_cardinality_and_order_are_exact():
    count = score(r"</think>\boxed{1},\boxed{2}", "1 and 2 and 3")
    assert count["grade"] == "incorrect"
    assert count["decision_reason"] == "collection_cardinality_mismatch"

    ordered = score(r"</think>\boxed{(2,1)}", "(1,2)")
    assert ordered["grade"] == "incorrect"
