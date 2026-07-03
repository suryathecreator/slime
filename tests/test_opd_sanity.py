from types import SimpleNamespace

import pytest

from slime.rollout.on_policy_distillation import maybe_check_rollout_sanity, summarize_rollout_sanity
from slime.utils.types import Sample


def _sample(response, *, length=100, status=Sample.Status.COMPLETED, rollout_id=0):
    return Sample(
        index=0,
        rollout_id=rollout_id,
        response=response,
        response_length=length,
        status=status,
    )


def test_opd_sanity_counts_cap_hits_and_final_answers():
    args = SimpleNamespace(rollout_max_response_len=1000)
    samples = [
        _sample("Reasoning... final answer is 7.", length=100),
        _sample("Reasoning... \\boxed{12}", length=200),
        _sample("unfinished", length=1000, status=Sample.Status.TRUNCATED),
    ]

    summary = summarize_rollout_sanity(args, samples)

    assert summary["n"] == 3
    assert summary["rollout_id"] == 0
    assert summary["cap_hit_rate"] == pytest.approx(1 / 3)
    assert summary["final_answer_rate"] == pytest.approx(2 / 3)


def test_opd_sanity_can_fail_on_collapse(monkeypatch):
    args = SimpleNamespace(rollout_max_response_len=1000)
    samples = [_sample("unfinished", length=1000, status=Sample.Status.TRUNCATED) for _ in range(4)]

    monkeypatch.setenv("OPD_SANITY_CHECK_ENABLED", "1")
    monkeypatch.setenv("OPD_SANITY_FAIL_ON_COLLAPSE", "1")
    monkeypatch.setenv("OPD_SANITY_MAX_ROLLOUT_ID", "0")
    monkeypatch.setenv("OPD_SANITY_MAX_CAP_HIT_RATE", "0.50")
    monkeypatch.setenv("OPD_SANITY_MIN_FINAL_ANSWER_RATE", "0.02")

    with pytest.raises(RuntimeError, match="OPD sanity check failed"):
        maybe_check_rollout_sanity(args, samples)
