"""CPU tests for the Qwen3-0.6B masked-SFT experiment contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    chat_prefix_suffix_ids,
)
from examples.qwen3_0_6b_openr1_math220k_masked_sft_2k.canary_generate import (
    normalize_generation_inputs,
    summarize_copy_regressions,
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


class FakeTensor:
    def __init__(self, shape=(1, 3)):
        self.shape = shape
        self.device = None

    def to(self, device):
        self.device = device
        return self


@pytest.mark.unit
def test_qwen3_chat_kwargs_are_frozen_into_pretokenization():
    prefix, suffix = chat_prefix_suffix_ids(
        FakeTokenizer(), "prompt", {"enable_thinking": True}
    )
    assert prefix == [151644, 77091, 198]
    assert suffix == [151645, 198]


@pytest.mark.unit
def test_generation_inputs_accept_raw_tensor():
    tensor = FakeTensor()
    assert normalize_generation_inputs(tensor, "cuda") == {"input_ids": tensor}
    assert tensor.device == "cuda"


@pytest.mark.unit
def test_generation_inputs_accept_mapping_and_forward_attention_mask():
    input_ids = FakeTensor()
    attention_mask = FakeTensor()
    unused = FakeTensor()
    normalized = normalize_generation_inputs(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": unused,
        },
        "cuda",
    )
    assert normalized == {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    assert input_ids.device == "cuda"
    assert attention_mask.device == "cuda"
    assert unused.device is None


@pytest.mark.unit
def test_generation_inputs_reject_mapping_without_input_ids():
    with pytest.raises(ValueError, match="missing input_ids"):
        normalize_generation_inputs({"attention_mask": FakeTensor()}, "cuda")


@pytest.mark.unit
def test_copy_regression_is_recorded_but_not_a_generation_failure():
    results = {
        "base": {"role_copy_starts": 0, "prompt_copy_starts": 1},
        "custom_pretokenized_after_two_updates": {
            "role_copy_starts": 2,
            "prompt_copy_starts": 1,
        },
        "stock_messages_after_two_updates": {
            "role_copy_starts": 0,
            "prompt_copy_starts": 3,
        },
    }
    diagnostic = summarize_copy_regressions(results)
    assert diagnostic["detected"] is True
    assert diagnostic["policy"] == "diagnostic_only_does_not_block_training"
    assert [item["regression"] for item in diagnostic["comparisons"]] == [
        True,
        False,
        False,
        True,
    ]


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
    assert "--branch qwen3-0.6b-openr1-masked-sft-2k" in submit
    assert "custom_pretokenized" in canary
    assert "slime.rollout.sft_rollout.generate_rollout" in canary
    assert "audit_runtime_batches.py" in canary
    assert "canary_generate.py" in canary
    assert 'SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000001"' in canary


@pytest.mark.unit
def test_generation_only_repair_reuses_artifacts_and_rewires_one_edge():
    generation_job = (EXAMPLE / "02_canary_generation_only.sbatch").read_text()
    repair = (EXAMPLE / "resubmit_after_canary_generation.sh").read_text()
    assert "#SBATCH --gres=gpu:h200:1" in generation_job
    assert "#SBATCH --time=01:00:00" in generation_job
    assert "canary_generate.py" in generation_job
    assert "diagnostic_only_does_not_block_training" in generation_job
    assert '[[ ! -e "${CANARY_GENERATION_JSON}" ]]' in generation_job
    for job_id in (207511, 207512, 207513, 207514, 207515, 207525):
        assert str(job_id) in repair
    assert "list(range(207511, 207526))" in repair
    for digest in (
        "114ef55784275913daca9412a05c3aec98f5d87301498a5b784c9ddcf0ebca90",
        "04922866b39dddd21311c4a7c5f6ed1ce7a9ef5147bfc5a22afe92ad5b3e7b83",
        "28105350fd5c29f29ecfced5d454a570f8cc9ca2fe1d1eba6760ae676181dbe3",
        "8df4bd72d116ba9c91e6653ff911f9aea23315e5cf2c1d40e61f4ab99456b925",
    ):
        assert digest in repair
    assert 'JobId="${BLOCKED_SCORE_JOB}"' in repair
    assert 'Dependency="afterok:${replacement_job}"' in repair
    assert '"reused_jobs": list(range(207514, 207526))' in repair


@pytest.mark.unit
def test_submission_manifest_accepts_explicit_branch_provenance(tmp_path):
    output = tmp_path / "manifest.json"
    writer = (
        ROOT
        / "examples/qwen2_5_7b_openr1_math220k_masked_sft_2k/write_submission_manifest.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(writer),
            "--output",
            str(output),
            "--contract",
            "contract",
            "--branch",
            "qwen3-branch",
            "--commit",
            "deadbeef",
            "--submitted-at",
            "2026-08-04T05:06:50+00:00",
            "model=1",
        ],
        check=True,
    )
    manifest = json.loads(output.read_text())
    assert manifest["branch"] == "qwen3-branch"
    assert manifest["commit"] == "deadbeef"
    assert manifest["submitted_at"] == "2026-08-04T05:06:50+00:00"


@pytest.mark.unit
def test_full_training_finishes_at_update_125_and_has_no_eval_stage():
    train = (EXAMPLE / "04_train.sbatch").read_text()
    submit = (EXAMPLE / "submit_training_only.sh").read_text()
    assert 'SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000124"' in train
    assert "eval" not in "\n".join(
        line for line in submit.splitlines() if line.lstrip().startswith("submit_serial")
    ).lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
