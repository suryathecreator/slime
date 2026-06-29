from argparse import Namespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module

NUM_GPUS = 0


def _make_args(entropy_coef):
    return Namespace(
        advantage_estimator="grpo",
        calculate_per_token_loss=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=entropy_coef,
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        rollout_top_p=1.0,
        use_kl_loss=False,
        use_opsm=False,
        use_rollout_logprobs=False,
        use_tis=False,
    )


def _make_batch():
    return {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([-0.4, -0.3])],
        "loss_masks": [torch.ones(2)],
        "response_lengths": [2],
        "total_lengths": [3],
        "unconcat_tokens": [torch.tensor([0, 1, 2])],
    }


def _sum_of_sample_mean(values):
    return values.mean()


@pytest.mark.unit
def test_policy_loss_skips_entropy_allocation_when_coef_is_zero(monkeypatch):
    seen = {}

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        seen["with_entropy"] = kwargs["with_entropy"]
        return None, {"log_probs": [torch.tensor([-0.2, -0.1])]}

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)

    _, report = loss_module.policy_loss_function(
        _make_args(entropy_coef=0.0),
        _make_batch(),
        torch.zeros(1, 3, 4),
        _sum_of_sample_mean,
    )

    assert seen["with_entropy"] is False
    torch.testing.assert_close(report["entropy_loss"], torch.tensor(0.0))


@pytest.mark.unit
def test_policy_loss_keeps_entropy_when_coef_is_nonzero(monkeypatch):
    seen = {}

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        seen["with_entropy"] = kwargs["with_entropy"]
        return None, {
            "log_probs": [torch.tensor([-0.2, -0.1])],
            "entropy": [torch.tensor([0.5, 1.5])],
        }

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)

    _, report = loss_module.policy_loss_function(
        _make_args(entropy_coef=0.01),
        _make_batch(),
        torch.zeros(1, 3, 4),
        _sum_of_sample_mean,
    )

    assert seen["with_entropy"] is True
    torch.testing.assert_close(report["entropy_loss"], torch.tensor(1.0))
