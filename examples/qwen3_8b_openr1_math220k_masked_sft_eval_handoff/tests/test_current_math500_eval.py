from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qwen3_8b_openr1_math220k_masked_sft_eval_handoff import (
    current_math500_eval as current,
)


HANDOFF = Path(__file__).resolve().parents[1]
REPO_ROOT = HANDOFF.parents[1]


def test_current_inventory_is_base_plus_three_40k() -> None:
    inventory = current.load_current_inventory(REPO_ROOT)
    assert [item["id"] for item in inventory["checkpoints"]] == [
        "base_8b",
        "40k_correct_only",
        "40k_weighted_tau_0p20",
        "40k_unmasked",
    ]
    assert inventory["runtime"]["vllm"] == "0.10.2"
    assert inventory["scorer"]["version"] == (
        "math500_strict_boxed_units_scorer_v4"
    )


def test_current_generation_policy_exactly_matches_2k() -> None:
    inventory = current.load_current_inventory(REPO_ROOT)
    policy = current.require_generation_policy(REPO_ROOT, inventory)
    assert policy["prompt"]["user_content"] == (
        'instruction + "\\n\\nProblem:\\n" + problem'
    )
    assert policy["decoding"]["max_tokens"] == 32768
    assert policy["engine"]["max_model_len"] == 32768
    assert policy["engine"]["vllm_version"] == "0.10.2"
    assert policy["persistence"]["chunk_size"] == 32
    assert policy["persistence"]["layout"] == (
        "four_contiguous_125_problem_shards"
    )
    text = (HANDOFF / "run_current_h200.sbatch").read_text()
    for required in (
        "--max_tokens 32768",
        "--max_model_len 32768",
        "--chunk_size 32",
        "--max_num_seqs 8",
        "--gpu_memory_utilization 0.92",
    ):
        assert required in text
    assert "safety" not in text.lower()


def test_source_join_preserves_canonical_order() -> None:
    canonical = [
        {"problem_id": "a", "correct_answer": "1"},
        {"problem_id": "b", "correct_answer": "2"},
    ]
    predictions = [
        {
            "problem_id": "b",
            "correct_answer": "2",
            "finish_reason": "stop",
        },
        {
            "problem_id": "a",
            "correct_answer": "1",
            "finish_reason": "length",
        },
    ]
    raw = [
        {"problem_id": "b", "generation": "B", "finish_reason": "stop"},
        {"problem_id": "a", "generation": "A", "finish_reason": "length"},
    ]
    ordered_predictions, ordered_raw = current.validate_source_rows(
        canonical, predictions, raw, source="fixture"
    )
    assert [row["problem_id"] for row in ordered_predictions] == ["a", "b"]
    assert [row["generation"] for row in ordered_raw] == ["A", "B"]


def test_source_join_rejects_mutated_gold() -> None:
    canonical = [{"problem_id": "a", "correct_answer": "1"}]
    predictions = [
        {
            "problem_id": "a",
            "correct_answer": "changed",
            "finish_reason": "stop",
        }
    ]
    raw = [{"problem_id": "a", "generation": "A", "finish_reason": "stop"}]
    with pytest.raises(RuntimeError, match="gold answer mismatch"):
        current.validate_source_rows(
            canonical, predictions, raw, source="fixture"
        )


def test_rescore_policy_and_readme_document_only_the_differences() -> None:
    policy = json.loads(
        (HANDOFF / "slime_v4_rescore_policy.json").read_text()
    )
    assert policy["scorer"]["version"] == (
        "math500_strict_boxed_units_scorer_v4"
    )
    assert "recognized explicit presentation units" in policy["answer_units"]
    assert "verbatim" in policy["generated_text"]
    readme = (HANDOFF / "README.md").read_text()
    assert "problem before the boxed-answer instruction" in readme
    assert "requests `max_tokens=32768`" in readme
    assert "vLLM 0.24.0" in readme
    assert "vLLM 0.10.2" in readme
    assert "not compatibility modes" in readme


def test_submission_is_strictly_base_first() -> None:
    text = (HANDOFF / "submit_base_40k_current.sh").read_text()
    assert '--dependency="afterok:$preflight_job"' in text
    assert '--dependency="afterok:$base_array"' in text
    assert '--dependency="afterok:$base_finalize"' in text
    assert "--array=0-3" in text
    assert "EVAL_ID=base_8b" in text
