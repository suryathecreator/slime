"""Deterministic unit tests for the 2K masking policies."""

from __future__ import annotations

import math

import pytest

from examples.qwen3_8b_openr1_math220k_masked_sft_2k.build_variants import (
    wrong_weights,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_2k.data_utils import (
    protected_think_positions,
    stable_unit_interval,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_2k.prepare_from_axolotl import (
    canonical_trace_structure,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {
            "<think>": [7, 8],
            "</think>": [9, 10],
        }[text]


def test_stable_draw_matches_axolotl_sha256_definition():
    first = stable_unit_interval(42, "trace", 3)
    second = stable_unit_interval(42, "trace", 3)
    assert first == second
    assert 0.0 <= first < 1.0
    assert first != stable_unit_interval(42, "trace", 4)


def test_protected_think_positions_cover_literal_tag_tokens():
    positions = protected_think_positions(
        FakeTokenizer(), [7, 8, 100, 101, 9, 10, 102]
    )
    assert positions == [0, 1, 4, 5]


def test_canonical_empty_think_trace_is_preserved():
    assert canonical_trace_structure("<think></think>answer") == (True, "ok")
    assert canonical_trace_structure("<think></think>")[1] == "empty_post_think"
    assert canonical_trace_structure("<think>reason answer") == (
        True,
        "missing_close_preserved",
    )
    assert canonical_trace_structure("<think>x</think></think>answer") == (
        True,
        "duplicate_close_preserved",
    )


def test_regular_and_inverse_margin_semantics():
    margins = [-0.20, 0.0, 0.20]
    assert wrong_weights("margin_mask", margins, "t", 42) == [0.0, 0.0, 1.0]
    assert wrong_weights("inverse_tau_0p20", margins, "t", 42) == pytest.approx(
        [0.5, 1.0, 1.0]
    )
    assert wrong_weights("inverse_tau_0p05", margins, "t", 42) == pytest.approx(
        [0.2, 1.0, 1.0]
    )


def test_probability_ratio_uses_exp_negative_margin_as_mask_probability(
    monkeypatch,
):
    monkeypatch.setattr(
        "examples.qwen3_8b_openr1_math220k_masked_sft_2k.build_variants.stable_unit_interval",
        lambda seed, trace_id, index: 0.5,
    )
    values = wrong_weights(
        "prob_ratio_mask", [-1.0, 0.0, math.log(4.0)], "t", 42
    )
    assert values == [0.0, 0.0, 1.0]


def test_random_mask_probability_is_applied_to_wrong_tokens(
    monkeypatch,
):
    draws = iter([0.24, 0.25, 0.99])
    monkeypatch.setattr(
        "examples.qwen3_8b_openr1_math220k_masked_sft_2k.build_variants.stable_unit_interval",
        lambda seed, trace_id, index: next(draws),
    )
    assert wrong_weights("random_mask_25", [0.0, 0.0, 0.0], "t", 42) == [
        0.0,
        1.0,
        1.0,
    ]
