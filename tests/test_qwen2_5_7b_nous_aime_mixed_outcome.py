from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from examples.qwen2_5_7b_nous_aime_mixed_outcome_sft.data_utils import (
    RUNTIME_PRESENTATIONS,
    VARIANTS,
    conditionally_uniform_subset,
    expand_balanced,
    select_conditioned_problem_subset,
)

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "examples/qwen2_5_7b_nous_aime_mixed_outcome_sft"
CONTRACT = json.loads((PACKAGE / "config/experiment_contract.json").read_text())


def source_record(index: int, doc_count: int, correct: bool) -> dict:
    return {
        "input_ids": [100 + index, 151645, 198],
        "metadata": {
            "base_trace_id": f"source_{index:04d}",
            "doc_id": str(index % doc_count),
            "is_correct": correct,
            "loss_weights": [1.0, 0.0],
            "response_length": 2,
            "trace_id": f"source_{index:04d}",
        },
    }


@pytest.mark.parametrize(
    "variant",
    ["aime24_correct_3000", "aime25_correct_3000"],
)
def test_expansion_balances_physical_and_runtime_exposure(
    variant: str,
) -> None:
    records = [source_record(index, 10, True) for index in range(181)]
    expanded, report = expand_balanced(records, variant=variant, seed=42)
    assert len(expanded) == 3000
    assert report["effective_runtime_presentations"] == RUNTIME_PRESENTATIONS
    order = list(range(3000))
    random.Random(43).shuffle(order)
    exposure = Counter(row["metadata"]["base_trace_id"] for row in expanded)
    for index in order[:8]:
        exposure[expanded[index]["metadata"]["base_trace_id"]] += 1
    assert Counter(exposure.values()) == Counter({16: 69, 17: 112})
    assert max(exposure.values()) - min(exposure.values()) == 1


def test_correct_subset_is_deterministic_and_covers_exact_problem_set() -> None:
    records = [source_record(index, 10, True) for index in range(456)]
    first, attempt = conditionally_uniform_subset(
        records,
        count=181,
        required_doc_ids={str(index) for index in range(10)},
        seed=42,
        year="aime24",
    )
    second, second_attempt = conditionally_uniform_subset(
        records,
        count=181,
        required_doc_ids={str(index) for index in range(10)},
        seed=42,
        year="aime24",
    )
    assert attempt == second_attempt
    assert [row["metadata"]["base_trace_id"] for row in first] == [row["metadata"]["base_trace_id"] for row in second]
    assert {row["metadata"]["doc_id"] for row in first} == {str(index) for index in range(10)}


def test_conditioned_problem_subset_is_deterministic_and_exact() -> None:
    incorrect_counts = {
        "1": 1,
        "6": 12,
        "8": 6,
        "9": 20,
        "10": 17,
        "11": 7,
        "12": 52,
        "18": 1,
        "19": 26,
        "20": 1,
        "21": 10,
        "22": 2,
        "23": 1,
        "26": 1,
        "27": 45,
        "28": 5,
        "29": 61,
    }
    selected, report = select_conditioned_problem_subset(
        incorrect_counts,
        problem_count=10,
        target_incorrect_rows=181,
        epsilon=0,
        seed=42,
        year="aime25",
    )
    assert selected == {"6", "8", "10", "19", "20", "21", "22", "26", "27", "29"}
    assert report["accepted_candidate_rank_zero_based"] == 358
    assert report["candidates_evaluated"] == 359
    assert report["selected_incorrect_rows"] == 181
    assert report["selected_delta"] == 0
    assert report["candidate_space_size"] == 19448
    assert report["qualifying_candidate_count"] == 212


def test_conditioned_problem_subset_fails_when_exact_match_is_impossible() -> None:
    with pytest.raises(ValueError, match="closest="):
        select_conditioned_problem_subset(
            {str(index): 1 for index in range(10)},
            problem_count=5,
            target_incorrect_rows=181,
            epsilon=0,
            seed=42,
            year="synthetic",
        )


def test_contract_pins_recipe_selection_and_monte_carlo_semantics() -> None:
    assert tuple(CONTRACT["variants"]) == VARIANTS
    training = CONTRACT["training"]
    assert training["global_batch_size"] == 64
    assert training["optimizer_updates"] == 47
    assert training["final_iteration"] == 46
    assert training["max_sequence_length"] == 32768
    assert training["learning_rate"] == 5e-6
    assert training["min_learning_rate"] == 1e-6
    assert training["adam_beta1"] == 0.9
    assert training["adam_beta2"] == 0.95
    assert training["adam_epsilon"] == 1e-8
    assert training["weight_decay"] == 1e-4
    assert CONTRACT["evaluation_handoff"]["draws_per_prompt"] == 16
    assert "not pass@16" in CONTRACT["evaluation_handoff"]["aggregation"]
    subset = CONTRACT["selection"]["conditioned_problem_subset"]
    assert subset["problem_count"] == 10
    assert subset["incorrect_trace_epsilon"] == 0
    assert subset["seed"] == 42
    assert "exact incorrect-trace-count match" in CONTRACT["selection"]["rule"]
    for year in ("aime24", "aime25"):
        assert CONTRACT["selection"]["expected_post_rescore_diagnostic"][year]["incorrect_rows"] == 181
        assert len(CONTRACT["selection"]["expected_post_rescore_diagnostic"][year]["selected_mixed_doc_ids"]) == 10
    assert CONTRACT["selection"]["expected_post_rescore_diagnostic"]["policy"].startswith("diagnostic only")


def test_documentation_contains_exact_prompt_and_selection_rule() -> None:
    readme = (PACKAGE / "README.md").read_text()
    exact_query = (
        "Solve the following math problem efficiently and clearly.  The last line of your response "
        "should be of the following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$. "
        "I hope it is correct'"
    )
    assert exact_query in readme
    assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>" in readme
    assert "take **all** retained incorrect" in readme
    assert "epsilon is zero" in readme.lower()
    assert "359 (zero-based rank 358)" in readme
    assert "10 held-in" in readme
    assert "16 independent" in readme
    assert "not pass@16" in readme


def test_shared_runner_interfaces_preserve_old_default() -> None:
    train = (ROOT / "examples/qwen2_5_7b_aime_generalization_incorrect_sft/03_train.sbatch").read_text()
    canary = (ROOT / "examples/qwen2_5_7b_aime_generalization_incorrect_sft/02_stage1_canary.sbatch").read_text()
    manifest = (ROOT / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py").read_text()
    assert "SFT_EXPERIMENT_ENV_FILE:-${SCRIPT_DIR}/env.sh" in train
    assert "SFT_MANIFEST_FAMILY:-aime_generalization" in train
    assert "SFT_CANARY_VARIANT" in canary
    assert "nous_aime_mixed_outcome" in manifest


def test_submission_is_ten_serial_jobs_and_handoff_excludes_training_data() -> None:
    submitter = (PACKAGE / "submit_training_only.sh").read_text()
    assert 'echo "NOUS_AIME_SBATCH_TEST_ONLY_OK jobs=10"' in submitter
    assert '--dependency="${dependency}"' in submitter
    assert "AIME_EXPECTED_GIT_COMMIT=${submit_commit}" in submitter
    rsync = (PACKAGE / "rsync_to_klone.sh").read_text()
    assert "training_jsonl=0 optimizer_state=0" in rsync
    assert "authenticated_sessions=1" in rsync
    assert "${DATA_ROOT}/datasets" not in rsync


def test_sbatch_resources_and_independent_base_initialization() -> None:
    env = (PACKAGE / "env.sh").read_text()
    assert "unset SFT_INITIAL_HF_DIR" in env
    assert "SFT_NUM_EPOCH=1" in env
    assert "SFT_NUM_ROLLOUT=47" in env
    assert "SFT_FINAL_ROLLOUT_ID=46" in env
    train = (PACKAGE / "03_train.sbatch").read_text()
    canary = (PACKAGE / "02_canary.sbatch").read_text()
    assert "#SBATCH --gres=gpu:h200:4" in train
    assert "#SBATCH --time=04:00:00" in train
    assert "#SBATCH --requeue" in train
    assert "#SBATCH --gres=gpu:h200:4" in canary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
