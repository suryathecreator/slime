"""Focused tests for deterministic masking and protected-token bypass."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants import wrong_weights
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    protected_assistant_positions,
    stable_unit_interval,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.prepare_from_axolotl import (
    canonical_trace_structure,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.weighted_sft_rollout import (
    generate_rollout,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.validate_base_model import (
    generation_prompt_ids,
)


class FakeTokenizer:
    def get_added_vocab(self):
        return {"<|control|>": 77}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"<think>": [7, 8], "</think>": [9, 10]}[text]


class FakeChatTokenizer:
    def __init__(self, rendered):
        self.rendered = rendered

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "x"}]
        assert tokenize is True
        assert add_generation_prompt is True
        return self.rendered


@pytest.mark.parametrize(
    "rendered",
    (
        [151644, 77091, 198],
        {"input_ids": [151644, 77091, 198], "attention_mask": [1, 1, 1]},
    ),
)
def test_generation_prompt_accepts_list_and_batch_encoding_shapes(rendered):
    assert generation_prompt_ids(FakeChatTokenizer(rendered)) == [
        151644,
        77091,
        198,
    ]


def test_stable_draw_is_repeatable():
    assert stable_unit_interval(42, "trace", 3) == stable_unit_interval(42, "trace", 3)
    assert stable_unit_interval(42, "trace", 3) != stable_unit_interval(42, "trace", 4)


def test_all_added_controls_and_think_pieces_are_protected():
    positions, reasons, literal_positions = protected_assistant_positions(
        FakeTokenizer(), [7, 8, 11, 77, 9, 10, 77]
    )
    assert positions == [0, 1, 3, 4, 5, 6]
    assert reasons["3"] == ["added_token:<|control|>"]
    assert reasons["0"] == ["literal_markup:<think>"]
    assert literal_positions == {"<think>": [0, 1], "</think>": [4, 5]}


@pytest.mark.parametrize(
    "variant",
    [
        "random_mask_25",
        "inverse_tau_0p20",
        "inverse_tau_0p05",
        "margin_mask",
        "prob_ratio_mask",
    ],
)
def test_protected_positions_bypass_every_masking_policy(monkeypatch, variant):
    def draw(seed, trace_id, index):
        if index in {0, 2}:
            raise AssertionError("protected position entered random policy")
        return 0.0

    monkeypatch.setattr(
        "examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants.stable_unit_interval",
        draw,
    )
    values = wrong_weights(variant, [-10.0, -10.0, -10.0], "t", 42, [0, 2])
    assert values[0] == 1.0
    assert values[2] == 1.0


def test_unprotected_margin_semantics():
    margins = [-0.20, 0.0, 0.20]
    assert wrong_weights("margin_mask", margins, "t", 42, []) == [0.0, 0.0, 1.0]
    assert wrong_weights("inverse_tau_0p20", margins, "t", 42, []) == pytest.approx(
        [0.5, 1.0, 1.0]
    )
    assert wrong_weights("inverse_tau_0p05", margins, "t", 42, []) == pytest.approx(
        [0.2, 1.0, 1.0]
    )


def test_probability_ratio_semantics(monkeypatch):
    monkeypatch.setattr(
        "examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants.stable_unit_interval",
        lambda seed, trace_id, index: 0.5,
    )
    assert wrong_weights(
        "prob_ratio_mask", [-1.0, 0.0, math.log(4.0)], "t", 42, []
    ) == [0.0, 0.0, 1.0]


def test_frozen_trace_anomalies_are_preserved_but_chat_controls_rejected():
    assert canonical_trace_structure("<think>x") == (True, "missing_close_preserved")
    assert canonical_trace_structure("<think>x</think></think>answer") == (
        True,
        "duplicate_close_preserved",
    )
    assert canonical_trace_structure("<think>x</think><|im_end|>answer") == (
        False,
        "embedded_chat_control",
    )


def test_rollout_rejects_a_downweighted_protected_control():
    sample = SimpleNamespace(
        prompt=[1, 2, 3, 151645, 198],
        metadata={
            "assistant_token_count": 3,
            "loss_weights": [1.0, 0.5, 1.0, 1.0, 0.0],
            "protected_assistant_positions": [1],
            "response_length": 5,
        },
    )
    buffer = SimpleNamespace(get_samples=lambda count: [[sample]])
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)
    with pytest.raises(ValueError, match="protected assistant controls"):
        generate_rollout(args, 0, buffer)


def test_model_validation_repair_reuses_the_pending_chain():
    root = Path(__file__).resolve().parents[1]
    text = (root / "resubmit_after_model_validation.sh").read_text()
    assert "readonly FAILED_MODEL_JOB=198094" in text
    assert "readonly BLOCKED_DATA_JOB=198095" in text
    assert "readonly NEXT_SMOKE_JOB=198096" in text
    assert "readonly TAIL_TRAIN_JOB=198108" in text
    assert 'scontrol update \\\n  JobId="${BLOCKED_DATA_JOB}"' in text
    assert 'Dependency="afterok:${replacement_job}"' in text
    assert '"reused_jobs": list(range(198095, 198109))' in text
    scancel_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("scancel")
    ]
    assert scancel_lines == ['      scancel "${replacement_job}" || true']
