from __future__ import annotations

from examples.qwen3_8b_32b_full_repro.audit_opd_rollouts import audit_sample, repetition_indicator


def sample() -> dict:
    return {
        "index": 0,
        "group_index": 0,
        "tokens": [100, 101, 151645],
        "response_length": 2,
        "response": r"<think>x</think>The answer is \boxed{2}.",
        "rollout_log_probs": [-0.4, -0.2],
        "reward": {
            "meta_info": {
                "input_token_logprobs": [
                    [None, 100, None],
                    [-1.0, 101, None],
                    [-0.5, 151645, None],
                ]
            }
        },
        "status": "completed",
        "metadata": {"source_row_id": 99, "prompt_sha256": "abc", "source": "math", "domain": "algebra"},
    }


def test_global_id_uses_root_rollout_and_local_order(monkeypatch) -> None:
    monkeypatch.setenv("OPD_GLOBAL_BATCH_SIZE", "64")
    monkeypatch.setenv("OPD_MAX_RESPONSE_LEN", "16384")
    first = audit_sample(3, 5, sample())
    second = audit_sample(4, 5, sample())
    assert first["global_trajectory_id"] == 197
    assert second["global_trajectory_id"] == 261
    assert first["global_trajectory_id"] != second["global_trajectory_id"]


def test_terminal_teacher_student_logprobs_are_aligned(monkeypatch) -> None:
    monkeypatch.setenv("OPD_GLOBAL_BATCH_SIZE", "64")
    monkeypatch.setenv("OPD_MAX_RESPONSE_LEN", "16384")
    row = audit_sample(0, 0, sample())
    assert row["terminal_condition"] == "im_end"
    assert row["teacher_terminal_logprob"] == -0.5
    assert row["student_terminal_logprob"] == -0.2
    assert row["aligned_logprob_tokens"] == 2


def test_repetition_indicator_is_bounded_and_deterministic() -> None:
    repeated = "\n".join(["the same sufficiently long repeated reasoning line"] * 4)
    assert repetition_indicator(repeated)
    assert not repetition_indicator("short unique reasoning")
