from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.eval import split_aime_eval
from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.eval.generate_split_aime import (
    DECODING as GENERATE_DECODING,
    EVAL_CONTRACT_SHA256 as GENERATE_CONTRACT_SHA256,
    PROMPTS as GENERATE_PROMPTS,
    SPLIT_SHA256 as GENERATE_SPLIT_SHA256,
    response_budget,
)
from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.eval.generate_split_aime import (
    sample_seed as generate_sample_seed,
)
from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.eval.split_aime_eval import (
    CONTRACT_ID,
    DECODING,
    DRAWS,
    PROMPTS,
    PROMPTS_PER_SPLIT,
    SPLITS,
    SPLIT_SHA256,
    TARGETS,
    draw_se,
    metrics,
    normalized_rows,
    paired_delta,
    parse_assignments,
    result_root,
    sample_seed,
    scorer_function,
    slice_selectors,
    target_paths,
    validate_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = REPO_ROOT / "checkpoints" / "qwen2_5_7b_aime_2009_2024_split_sft_6k" / "v1" / CONTRACT_ID
SIXTEEN_X_POLICY = REPO_ROOT / "examples/qwen3_8b_aime_2009_2024_16x/eval_policy.json"
EVAL_POLICY = Path(__file__).resolve().parents[1] / "eval_policy.json"


def args() -> argparse.Namespace:
    return argparse.Namespace(repo_root=REPO_ROOT, experiment_root=EXPERIMENT_ROOT)


requires_transfer = pytest.mark.skipif(
    not EXPERIMENT_ROOT.is_dir(), reason="split-control checkpoints are not transferred here"
)


def test_shape_constants_match_the_four_draw_contract() -> None:
    assert DRAWS == 4
    assert SPLITS == ("held_in", "held_out")
    assert PROMPTS_PER_SPLIT == 106
    assert PROMPTS == 212
    assert len(TARGETS) == 5
    assert PROMPTS * DRAWS * len(TARGETS) == 4240


def test_sampling_is_reused_verbatim_from_the_16x_run_except_the_draw_count() -> None:
    source = json.loads(SIXTEEN_X_POLICY.read_text(encoding="utf-8"))["generation"]
    for field in ("temperature", "top_p", "top_k", "min_p"):
        assert DECODING[field] == source[field], field
    assert source["completions_per_problem"] == 16
    policy = json.loads(EVAL_POLICY.read_text(encoding="utf-8"))
    assert policy["generation"]["completions_per_problem"] == DRAWS
    assert policy["generation"]["seed_formula"] == source["seed_formula"]
    assert policy["generation"]["seed_base"] == source["seed_base"]


def test_generation_and_control_agree_on_policy() -> None:
    assert GENERATE_DECODING == DECODING
    assert GENERATE_SPLIT_SHA256 == SPLIT_SHA256
    assert GENERATE_CONTRACT_SHA256 == split_aime_eval.EVAL_CONTRACT_SHA256
    assert GENERATE_PROMPTS == PROMPTS_PER_SPLIT


def test_seed_is_collision_free_across_draws_and_both_splits() -> None:
    # eval_index is inherited from the 480-problem corpus and unique across splits.
    seeds = {sample_seed(draw, index) for draw in range(DRAWS) for index in range(480)}
    assert len(seeds) == DRAWS * 480
    assert sample_seed(0, 0) == 1234
    assert sample_seed(1, 0) == 1234 + 480
    assert generate_sample_seed(3, 479) == sample_seed(3, 479)
    with pytest.raises(ValueError):
        sample_seed(DRAWS, 0)
    with pytest.raises(ValueError):
        sample_seed(0, 480)


def test_response_budget_rejects_a_prompt_that_fills_the_context() -> None:
    assert response_budget(768) == 32000
    with pytest.raises(RuntimeError):
        response_budget(32768)


def test_results_are_written_to_scratch_not_the_package() -> None:
    # The runtime repo gate requires a clean worktree on every delayed job.
    root = result_root(args())
    assert REPO_ROOT / "examples" not in root.parents
    assert root.is_relative_to(EXPERIMENT_ROOT)


def test_slices_cover_both_splits_disjointly() -> None:
    selectors = slice_selectors()
    assert set(selectors) == {"held_in", "held_out", "all_212"}
    row = {"split": "held_in"}
    assert selectors["held_in"](row) and not selectors["held_out"](row)
    assert selectors["all_212"](row)


def test_parse_assignments_requires_every_target_exactly_once() -> None:
    good = [f"{target}=100{index}" for index, target in enumerate(TARGETS)]
    assert set(parse_assignments(good)) == set(TARGETS)
    with pytest.raises(RuntimeError):
        parse_assignments(good[:-1])
    with pytest.raises(RuntimeError):
        parse_assignments([*good, good[0]])
    with pytest.raises(RuntimeError):
        parse_assignments([f"{TARGETS[0]}=notanumber", *good[1:]])


def scored(draw: int, index: int, correct: bool, split: str = "held_in") -> dict[str, object]:
    return {
        "cap_hit": False,
        "draw_index": draw,
        "eval_index": index,
        "generated_token_count": 10,
        "is_correct": correct,
        "is_scorable": True,
        "split": split,
    }


def test_draw_se_measures_spread_across_draws_not_completions() -> None:
    identical = [scored(draw, index, True) for draw in range(DRAWS) for index in range(2)]
    assert draw_se(identical) == pytest.approx(0.0)
    split_rows = [scored(draw, index, draw == 0) for draw in range(DRAWS) for index in range(2)]
    assert draw_se(split_rows) > 0.0
    assert metrics(identical)["draws"] == DRAWS


def test_paired_delta_requires_matching_coordinates() -> None:
    left = [scored(0, 0, False), scored(0, 1, False)]
    right = [scored(0, 0, True), scored(0, 1, False)]
    assert paired_delta(left, right)["delta"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError):
        paired_delta(left, [scored(0, 0, True), scored(0, 2, False)])


def test_scorer_identity_is_pinned() -> None:
    score_response = scorer_function(args())
    trace = score_response(
        "the answer is \\boxed{042}",
        42,
        finish_reason="stop",
        stop_token_id=151645,
        cap_hit=False,
        problem_id="2011-I-1",
    )
    assert trace["is_correct"] is True
    assert trace["scorer_version"] == "aime_last_boxed_integer_scorer_v1"


def test_multiple_accepted_answers_are_both_credited() -> None:
    score_response = scorer_function(args())
    record = {
        "response": "so the answer is \\boxed{081}",
        "finish_reason": "stop",
        "stop_reason": 151645,
        "cap_hit": False,
        "problem_id": "2022-II-8",
    }
    assert split_aime_eval.score_accepted(score_response, record, [80, 81])["is_correct"] is True
    assert split_aime_eval.score_accepted(score_response, record, [80])["is_correct"] is False
    # A non-matching response still yields a well-defined trace.
    miss = dict(record, response="\\boxed{123}")
    assert split_aime_eval.score_accepted(score_response, miss, [80, 81])["is_correct"] is False


@requires_transfer
def test_contract_and_eval_sets_match_the_pinned_hashes() -> None:
    contract = validate_contract(args())
    assert tuple(contract["checkpoints"]) == TARGETS
    assert contract["prompt"]["uses_qwen2_5_default_chat_template"] is False
    assert contract["cap_hits_scored"] is True


@requires_transfer
def test_rows_use_verbatim_prompts_with_no_system_turn() -> None:
    rows = normalized_rows(args())
    assert len(rows) == PROMPTS
    assert len({row["eval_index"] for row in rows}) == PROMPTS
    for row in rows:
        rendered = row["rendered_prompt"]
        assert rendered.startswith("<|im_start|>user\n")
        assert rendered.endswith("<|im_end|>\n<|im_start|>assistant\n")
        assert "<|im_start|>system" not in rendered
        assert row["prompt"] in rendered
    assert sum(len(row["accepted_answer_integers"]) > 1 for row in rows) == 1


@requires_transfer
def test_every_target_resolves_to_a_transferred_checkpoint() -> None:
    for target in TARGETS:
        model, manifest, identity = target_paths(args(), target)
        assert model.is_dir(), target
        assert manifest.is_file(), target
        assert len(identity) == 64
