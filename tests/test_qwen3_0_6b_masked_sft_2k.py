"""CPU tests for the Qwen3-0.6B masked-SFT experiment contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    chat_prefix_suffix_ids,
)
from examples.qwen3_0_6b_openr1_math220k_masked_sft_2k.validate_schedule import (
    audit,
)


NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/qwen3_0_6b_openr1_math220k_masked_sft_2k"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "prompt"}]
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
        return [151644, 77091, 198]

    def encode(self, text, add_special_tokens=False):
        assert text == "<|im_end|>\n"
        assert add_special_tokens is False
        return [151645, 198]


@pytest.mark.unit
def test_qwen3_chat_kwargs_are_frozen_into_pretokenization():
    prefix, suffix = chat_prefix_suffix_ids(
        FakeTokenizer(), "prompt", {"enable_thinking": True}
    )
    assert prefix == [151644, 77091, 198]
    assert suffix == [151645, 198]


@pytest.mark.unit
def test_contract_locks_requested_hyperparameters():
    contract = json.loads((EXAMPLE / "config/experiment_contract.json").read_text())
    training = contract["training"]
    assert training == {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "context_length": 32768,
        "data_parallel_size": 4,
        "effective_gradient_accumulation_steps": 1,
        "epochs": 1,
        "full_parameter_sft": True,
        "global_batch_size": 16,
        "gradient_clip_norm": 1.0,
        "learning_rate": 1e-5,
        "lr_decay_style": "cosine",
        "max_tokens_per_gpu": 32768,
        "min_learning_rate": 0.0,
        "optimizer": "AdamW",
        "optimizer_updates": 125,
        "rows": 2000,
        "tensor_parallel_size": 1,
        "warmup_fraction": 0.05,
        "warmup_style": "linear_from_zero",
        "weight_decay": 1e-4,
    }
    assert contract["base_model"]["revision"] == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


@pytest.mark.unit
def test_dynamic_schedule_proves_gradient_accumulation_one(tmp_path):
    data = tmp_path / "records.jsonl"
    with data.open("w", encoding="utf-8") as handle:
        for index in range(2000):
            length = 100 + index % 300
            handle.write(json.dumps({"input_ids": [0] * length}) + "\n")
    result = audit(data, seed=42, batch_size=16, dp_size=4, max_tokens=32768)
    assert result["optimizer_updates"] == 125
    assert result["effective_gradient_accumulation_steps"] == 1
    assert sum(result["rows_per_rank"]) == 2000


@pytest.mark.unit
def test_shared_runner_forwards_exact_adamw_schedule_knobs():
    runner = (ROOT / "examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch").read_text()
    container = (ROOT / "examples/qwen3_8b_opd_tillicum/container_exec.sh").read_text()
    for needle in (
        '--lr-decay-style "${SFT_LR_DECAY_STYLE}"',
        '--lr-warmup-fraction "${SFT_LR_WARMUP_FRACTION}"',
        '--lr-decay-iters "${SFT_LR_DECAY_ITERS}"',
        '--weight-decay "${SFT_WEIGHT_DECAY}"',
        '--adam-beta1 "${SFT_ADAM_BETA1}"',
        '--adam-beta2 "${SFT_ADAM_BETA2}"',
        '--adam-eps "${SFT_ADAM_EPS}"',
    ):
        assert needle in runner
    for name in (
        "SFT_LR_DECAY_STYLE",
        "SFT_LR_WARMUP_FRACTION",
        "SFT_LR_DECAY_ITERS",
        "SFT_WEIGHT_DECAY",
        "SFT_ADAM_EPS",
    ):
        assert name in container


@pytest.mark.unit
def test_canary_gates_the_serial_training_chain():
    submit = (EXAMPLE / "submit_training_only.sh").read_text()
    canary = (EXAMPLE / "02_canary.sbatch").read_text()
    assert submit.index("canary_qwen3_0p6b_custom_and_stock") < submit.index(
        "score_qwen3_0p6b_margins_and_build"
    )
    assert "afterok:${tail_job}" in submit
    assert "custom_pretokenized" in canary
    assert "slime.rollout.sft_rollout.generate_rollout" in canary
    assert "audit_runtime_batches.py" in canary
    assert "canary_generate.py" in canary
    assert 'SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000001"' in canary


@pytest.mark.unit
def test_full_training_finishes_at_update_125_and_has_no_eval_stage():
    train = (EXAMPLE / "04_train.sbatch").read_text()
    submit = (EXAMPLE / "submit_training_only.sh").read_text()
    assert 'SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000124"' in train
    assert "eval" not in "\n".join(
        line for line in submit.splitlines() if line.lstrip().startswith("submit_serial")
    ).lower()
