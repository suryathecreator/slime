from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.data_utils import (
    DATASET_ROWS,
    RUNTIME_PRESENTATIONS,
    UPDATES,
    VARIANTS,
    expand_balanced,
    partition_problem_ids,
    rendered_user_prefix,
    select_correct_control,
)
from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.finalize import parse_metrics

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "examples/qwen2_5_7b_aime_2009_2024_split_sft_6k"
CONTRACT = json.loads((PACKAGE / "config/experiment_contract.json").read_text())


def source_record(index: int, problem_id: str, correct: bool = True) -> dict:
    return {
        "input_ids": [100 + index, 151645, 198],
        "metadata": {
            "base_trace_id": f"source_{index:04d}",
            "is_correct": correct,
            "problem_id": problem_id,
            "response_length": 2,
            "trace_id": f"source_{index:04d}",
        },
    }


def test_problem_partition_is_deterministic_disjoint_and_exhaustive() -> None:
    problem_ids = [f"problem-{index:03d}" for index in range(480)]
    first_in, first_out = partition_problem_ids(problem_ids, held_in_count=240, seed=42)
    second_in, second_out = partition_problem_ids(reversed(problem_ids), held_in_count=240, seed=42)
    assert first_in == second_in
    assert first_out == second_out
    assert len(first_in) == len(first_out) == 240
    assert not first_in & first_out
    assert first_in | first_out == set(problem_ids)


def test_correct_control_matches_trace_and_problem_counts() -> None:
    records = [source_record(problem * 20 + draw, f"p{problem}") for problem in range(12) for draw in range(20)]
    selected, report = select_correct_control(
        records,
        target_rows=70,
        target_problem_count=7,
        seed=42,
        split="held_in",
    )
    assert len(selected) == 70
    assert len({row["metadata"]["problem_id"] for row in selected}) == 7
    assert report["selected_rows"] == 70
    assert report["selected_problem_count"] == 7
    repeated, repeated_report = select_correct_control(
        list(reversed(records)),
        target_rows=70,
        target_problem_count=7,
        seed=42,
        split="held_in",
    )
    assert [row["metadata"]["base_trace_id"] for row in repeated] == [
        row["metadata"]["base_trace_id"] for row in selected
    ]
    assert repeated_report == report


def test_balanced_expansion_gives_every_trace_nearly_equal_exposure() -> None:
    records = [source_record(index, f"p{index % 31}") for index in range(731)]
    expanded, report = expand_balanced(records, variant=VARIANTS[0], seed=42)
    assert len(expanded) == DATASET_ROWS
    exposure = Counter(row["metadata"]["base_trace_id"] for row in expanded)
    assert max(exposure.values()) - min(exposure.values()) == 1
    assert report["dataset_rows"] == 6000
    assert report["three_full_epochs_presentations"] == 18000
    assert report["effective_runtime_presentations"] == 18048
    assert report["wrap_presentations"] == 48


def test_contract_pins_three_epoch_recipe_and_unfiltered_source() -> None:
    assert tuple(CONTRACT["variants"]) == VARIANTS
    training = CONTRACT["training"]
    assert training["epochs"] == 3
    assert training["optimizer_updates"] == UPDATES == 282
    assert training["final_iteration"] == 281
    assert training["effective_presentations"] == RUNTIME_PRESENTATIONS == 18048
    assert training["max_sequence_length"] == 32896
    assert training["global_batch_size"] == 64
    assert training["learning_rate"] == 5e-6
    assert training["min_learning_rate"] == 1e-6
    assert training["adam_beta1"] == 0.9
    assert training["adam_beta2"] == 0.95
    assert training["adam_epsilon"] == 1e-8
    assert training["weight_decay"] == 1e-4
    assert "No source trace is filtered or truncated" in CONTRACT["selection"]["length_policy"]
    audit = CONTRACT["selection"]["realized_source_audit"]
    assert audit["maximum_fully_rendered_tokens"] == 32774
    assert (audit["held_in"]["incorrect_rows"], audit["held_in"]["incorrect_problem_count"]) == (717, 123)
    assert (audit["held_out"]["incorrect_rows"], audit["held_out"]["incorrect_problem_count"]) == (814, 116)


def test_exact_prompt_has_one_user_turn_and_no_system() -> None:
    prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n2+2?"
    assert rendered_user_prefix(prompt) == ("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
    assert CONTRACT["formatting"]["system"] is None
    assert CONTRACT["formatting"]["uses_apply_chat_template"] is False
    readme = (PACKAGE / "README.md").read_text()
    assert "no system message" in readme.lower()
    assert "{SOURCE_RESPONSE_VERBATIM}<|im_end|>" in readme
    assert "need not use the exact same" in readme


def test_dag_is_ten_serial_jobs_and_uses_immutable_worktree() -> None:
    submitter = (PACKAGE / "submit_training_only.sh").read_text()
    assert 'echo "AIME_SPLIT_6K_SBATCH_TEST_ONLY_OK jobs=10' in submitter
    assert "git worktree add --detach" in submitter
    assert 'cd "${EXECUTION_WORKTREE}"' in submitter
    assert '--chdir="${EXECUTION_WORKTREE}"' in submitter
    assert '--dependency="${dependency}"' in submitter
    assert "AIME_EXPECTED_GIT_COMMIT=${submit_commit}" in submitter
    assert 'submit_job "canary_${variant}"' in submitter
    assert 'submit_job "train_${variant}"' in submitter


def test_sbatch_resources_and_independent_base_initialization() -> None:
    env = (PACKAGE / "env.sh").read_text()
    assert "unset SFT_INITIAL_HF_DIR" in env
    assert "SFT_SEQ_LENGTH=32896" in env
    assert "SFT_NUM_EPOCH=3" in env
    assert "AIME_SPLIT_6K_OPTIMIZER_UPDATES=282" in env
    assert "SFT_SAVE_INTERVAL=24" in env
    for filename, walltime, gpu_count in (
        ("01_prepare_data.sbatch", "02:00:00", 1),
        ("02_canary.sbatch", "02:00:00", 4),
        ("03_train.sbatch", "08:00:00", 4),
        ("05_finalize.sbatch", "02:00:00", 1),
    ):
        text = (PACKAGE / filename).read_text()
        assert f"#SBATCH --time={walltime}" in text
        assert f"#SBATCH --gres=gpu:h200:{gpu_count}" in text


def test_handoff_excludes_training_data_and_defers_eval_sampling() -> None:
    finalizer = (PACKAGE / "finalize.py").read_text()
    assert '"sampling": "deferred_to_Hyak_and_must_be_recorded"' in finalizer
    assert "data/eval/held_in_240.jsonl" in finalizer
    assert "data/eval/held_out_240.jsonl" in finalizer
    assert "data/datasets" not in finalizer
    rsync = (PACKAGE / "rsync_to_klone.sh").read_text()
    assert "training_jsonl=0 optimizer_state=0" in rsync
    assert "authenticated_sessions=1" in rsync
    assert "${DATA_ROOT}/datasets" not in rsync


def test_checkpoint_manifest_accepts_experiment_family() -> None:
    manifest = (ROOT / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py").read_text()
    assert '"aime_2009_2024_split"' in manifest


def test_source_revision_and_all_sixteen_hashes_are_pinned() -> None:
    source = CONTRACT["source_dataset"]
    assert source["hf_repo"] == "suryadv/qwen3-8b-aime-2009-2024-16x"
    assert source["revision"] == "c34143c92856d9b90fdb9c808c7e66070dcab9a1"
    assert set(source["files"]) == {f"draw-{index:02d}.parquet" for index in range(16)}
    assert all(len(value) == 64 for value in source["files"].values())
    assert "accepted_answer_integers" in (PACKAGE / "prepare_data.py").read_text()


def test_loss_parser_uses_surviving_last_occurrence_after_requeue(tmp_path: Path) -> None:
    def line(step: int, loss: float) -> str:
        return (
            f"step {step}: {{'train/loss': {loss}, 'train/global_batch_size': 64, "
            f"'train/grad_norm': 1.0, 'train/lr-pg_0': 1e-06, "
            f"'train/lr-pg_1': 1e-06, 'train/step': {step}}}\n"
        )

    log = tmp_path / "train.log"
    log.write_text(line(0, 2.0) + line(0, 1.5) + line(1, 1.0))
    history = parse_metrics(log, final_iteration=1)
    assert [point["loss"] for point in history] == [1.5, 1.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
